"""M3 thin CLI: global dedup + M1 downsample + stratified split + leakage check.

Usage:
    uv run python scripts/data/split.py --config configs/split.yaml \
        [--smoke] [--output-dir DIR]

Runtime order (the M3 contract, the M3 split target): merge M1+M2 -> global
exact dedup -> global MinHash near-dedup (BEFORE split) -> M1 downsample (T3.0)
-> stratified split by complete dialogue -> cross-split leakage assertion.

Writes ``{train,val,test}.jsonl`` + ``split_report.json`` (section 2.3 contract)
+ ``manifest.json`` into the output directory. Exit codes: 0 success, 2 input
contract failure OR cross-split leakage detected (acceptance #3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl, write_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.common.schema import DialogueRecord, validate_dialogue
from sales_agent.data.dedup import dedup_exact
from sales_agent.data.split import (
    assert_no_leakage,
    default_stratum_key,
    distribution_warnings,
    downsample_m1,
    is_real,
    minhash_dedup,
    scenario_distribution,
    stratified_split,
)

logger = logging.getLogger("split")

EXIT_OK = 0
EXIT_CONTRACT = 2


def load_records(path: Path) -> tuple[list[DialogueRecord], list[str]]:
    """Parse a JSONL file into validated DialogueRecords; collect error strings."""
    records: list[DialogueRecord] = []
    errors: list[str] = []
    for i, raw in enumerate(read_jsonl(path)):
        try:
            rec = DialogueRecord.model_validate(raw)
        except Exception as e:  # pydantic structural failure
            errors.append(f"{path.name}:{i}: {e}")
            continue
        semantic = validate_dialogue(rec)
        if semantic:
            errors.append(f"{path.name}:{i}: {'; '.join(semantic)}")
            continue
        records.append(rec)
    return records, errors


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="cap records per input")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["output_dir"])
    seed = cfg["seed"]
    inputs = cfg.get("inputs", [])
    if not inputs:
        logger.error("config has no inputs")
        return EXIT_CONTRACT
    limit = cfg.get("smoke_limit", 300) if args.smoke else None

    # 1. Load + validate inputs.
    all_records: list[DialogueRecord] = []
    input_paths: list[str] = []
    for spec in inputs:
        path = Path(spec["path"])
        input_paths.append(str(path))
        try:
            recs, errors = load_records(path)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("cannot read input %s: %s", path, e)
            return EXIT_CONTRACT
        if errors:
            for e in errors[:10]:
                logger.error("invalid record: %s", e)
            logger.error("%s: %d invalid records (contract failure)", path.name, len(errors))
            return EXIT_CONTRACT
        if limit is not None:
            recs = recs[:limit]
        logger.info("loaded %d records from %s", len(recs), path.name)
        all_records.extend(recs)

    # 2. Global exact dedup, then global MinHash near-dedup (BEFORE split).
    dcfg = cfg.get("dedup", {})
    threshold = dcfg.get("threshold", 0.85)
    num_perm = dcfg.get("num_perm", 128)
    shingle_size = dcfg.get("shingle_size", 5)

    exact_kept, exact_dropped = dedup_exact(all_records)
    logger.info(
        "exact dedup: %d -> %d (dropped %d)", len(all_records), len(exact_kept), exact_dropped
    )
    survivors, dedup_stats = minhash_dedup(
        exact_kept, threshold=threshold, seed=seed, num_perm=num_perm, shingle_size=shingle_size
    )
    logger.info(
        "minhash dedup@%.2f: %d -> %d (dropped %d, clusters %d)",
        threshold, dedup_stats.n_input, dedup_stats.n_kept, dedup_stats.n_dropped,
        dedup_stats.n_clusters,
    )

    # 3. Split survivors into M1 (real) vs M2 (synthetic); downsample M1 only.
    m1_surv = [r for r in survivors if is_real(r)]
    m2_surv = [r for r in survivors if not is_real(r)]
    logger.info("after global dedup: M1=%d M2=%d", len(m1_surv), len(m2_surv))

    ds = cfg.get("downsample", {})
    m1_kept, ds_report = downsample_m1(
        m1_surv,
        m2_count=len(m2_surv),
        ratio=ds.get("ratio", 2.0),
        energy_keywords=ds.get("energy_keywords", []),
        high_value_scenarios=ds.get("high_value_scenarios", []),
        seed=seed,
    )
    logger.info(
        "downsample M1: target=%d union=%d (label_only=%d kw_only=%d both=%d) "
        "energy_hits=%d filled_general=%d -> final_M1=%d (M1:M2=%.2f)",
        ds_report.target_m1, ds_report.high_value_union, ds_report.label_only,
        ds_report.keyword_only, ds_report.both, ds_report.energy_keyword_hits,
        ds_report.filled_general, ds_report.final_m1, ds_report.m1_to_m2_ratio,
    )
    if ds_report.high_value_exceeds_target:
        logger.warning(
            "high-value union (%d) >= target (%d): kept ALL high-value, ratio breached upward. "
            "No high-value data discarded. Review the data mix.",
            ds_report.high_value_union, ds_report.target_m1,
        )

    # 4. Stratified split by complete dialogue.
    merged = m1_kept + m2_surv
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.9, 0.05, 0.05]))
    min_stratum = cfg.get("split", {}).get("min_stratum", 20)
    splits, split_meta = stratified_split(
        merged, ratios, default_stratum_key, seed=seed, min_stratum=min_stratum
    )
    logger.info(
        "split: train=%d val=%d test=%d (from %d merged)",
        len(splits["train"]), len(splits["val"]), len(splits["test"]), len(merged),
    )
    if not splits["val"] or not splits["test"]:
        logger.error(
            "empty val or test split (val=%d test=%d) -- stratification could not allocate; "
            "report to maintainer, do not silently proceed",
            len(splits["val"]), len(splits["test"]),
        )
        return EXIT_CONTRACT

    # 5. Distribution consistency (>3pp WARN) + leakage assertion.
    warn_pp = cfg.get("dist_warn_pp", 3.0)
    dist_warnings = distribution_warnings(splits, default_stratum_key, warn_pp=warn_pp)
    for w in dist_warnings:
        logger.warning(
            "distribution skew: %s stratum=%s split_share=%.3f overall=%.3f (%.2fpp)",
            w["split"], w["stratum"], w["split_share"], w["overall_share"], w["deviation_pp"],
        )

    leak = assert_no_leakage(
        splits, threshold=threshold, seed=seed, num_perm=num_perm, shingle_size=shingle_size
    )

    # 6. Assemble split_report.json (section 2.3 contract + M3 extras).
    report = {
        "counts": {k: len(v) for k, v in splits.items()},
        "ratios": list(ratios),
        "stratify_keys": ["scenario", "turn_bucket"],
        "distribution": {k: scenario_distribution(v) for k, v in splits.items()},
        "leakage_check": leak.as_dict(),
        "dedup": {
            "exact_dropped": exact_dropped,
            "minhash_threshold": threshold,
            "minhash_dropped": dedup_stats.n_dropped,
            "minhash_clusters": dedup_stats.n_clusters,
            "n_after_dedup": dedup_stats.n_kept,
        },
        "downsample": {
            **ds_report.as_dict(),
            # Deviation of the tightened \b matcher vs the user's original loose
            # substring probe (baseline computed on full pre-dedup M1). The loose
            # probe over-counted via generic words (provider/supplier/electric).
            "energy_probe_baseline": ds.get("energy_probe_baseline"),
            "energy_keyword_deviation_vs_probe": (
                ds_report.energy_keyword_hits - ds["energy_probe_baseline"]
                if ds.get("energy_probe_baseline") is not None
                else None
            ),
        },
        "effective_strata": split_meta.effective_strata,
        "merged_strata": split_meta.merged,
        "distribution_warnings": dist_warnings,
    }

    # 7. Fail (exit 2) on cross-split leakage -- acceptance #3.
    if leak.cross_split_dups > 0:
        logger.error(
            "LEAKAGE: %d cross-split near-duplicate pairs (must be 0). Examples: %s",
            leak.cross_split_dups, json.dumps(leak.examples[:5]),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "split_report.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return EXIT_CONTRACT

    # 8. Write splits, report, manifest.
    for name in ("train", "val", "test"):
        write_jsonl(output_dir / f"{name}.jsonl", (r.model_dump() for r in splits[name]))
    report_path = output_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    manifest = build_manifest(
        inputs=input_paths,
        config=cfg,
        stats={"counts": report["counts"], "leakage_check": report["leakage_check"]},
    )
    write_manifest(output_dir, manifest)

    logger.info(
        "wrote train/val/test -> %s | counts=%s | cross_split_dups=%d | report=%s",
        output_dir, report["counts"], leak.cross_split_dups, report_path,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
