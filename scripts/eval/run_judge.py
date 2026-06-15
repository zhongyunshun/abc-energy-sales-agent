"""M10 thin CLI: LLM-as-a-Judge scoring of the three model groups.

Usage (smoke first, then full -- after M9 produced the three results.jsonl):
    uv run python scripts/eval/run_judge.py --config configs/eval_judge.yaml --smoke
    uv run python scripts/eval/run_judge.py --config configs/eval_judge.yaml

Flow (design doc 3-M10): load base/sft/dpo results.jsonl -> select_judge_samples
(SAME ids across groups, seeded, scenario-stratified) -> run_judge (each non-Google
judge model scores every sample BLIND, via OpenRouter) -> aggregate_scores ->
write reports/eval_judge/{scores.jsonl, comparison.md, manifest.json}.

Exit codes (design doc 1.4): 0 success; 2 input contract failure (missing/empty/
malformed results.jsonl, no common ids); 3 external dependency failure (no API key,
or every judge call failed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from sales_agent.common.config import find_repo_root, load_config
from sales_agent.common.io import read_jsonl, write_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.common.openrouter import OpenRouterClient, OpenRouterError
from sales_agent.evals.judge import (
    JudgeConfig,
    aggregate_scores,
    estimate_cost,
    load_template,
    render_comparison_md,
    run_judge,
    select_judge_samples,
)

logger = logging.getLogger("run_judge")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_DEPENDENCY = 3


def _resolve_input_dirs(args_inputs, cfg_inputs) -> list[Path]:
    """CLI --inputs (absolute or repo-relative) override config 'inputs' (repo-relative)."""
    root = find_repo_root()
    raw = args_inputs if args_inputs else (cfg_inputs or [])
    dirs = []
    for d in raw:
        p = Path(d)
        dirs.append(p if p.is_absolute() else (root / p))
    return dirs


def _load_results(input_dirs: list[Path]) -> dict[str, list[dict]]:
    """Load each dir's results.jsonl into {tag(basename): rows}; raise on contract faults."""
    out: dict[str, list[dict]] = {}
    for d in input_dirs:
        path = d / "results.jsonl"
        tag = d.name
        if not path.exists():
            raise FileNotFoundError(f"missing results.jsonl for tag {tag!r}: {path}")
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"empty results.jsonl for tag {tag!r}: {path}")
        out[tag] = rows
    return out


def load_env_file(path: Path) -> None:
    """Minimal .env loader (no extra dependency), mirroring M2's synthesize CLI:
    KEY=VALUE lines into os.environ. Existing env vars win; quotes stripped."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_client(cfg: JudgeConfig) -> OpenRouterClient:
    """Construct the OpenRouter client (reads OPENROUTER_API_KEY). Patched in tests."""
    return OpenRouterClient(
        model=cfg.judge_models[0],
        concurrency=cfg.concurrency,
        max_retries=cfg.client_max_retries,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", nargs="+", default=None, help="model-group result dirs")
    parser.add_argument(
        "--judge-model", action="append", default=None,
        help="override config judge_models (repeatable)",
    )
    parser.add_argument("--smoke", action="store_true", help="use smoke.n_samples (5/group)")
    parser.add_argument("--n", type=int, default=None, help="override sampling.n_samples")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    cfg_raw = load_config(args.config)
    jcfg = JudgeConfig.from_dict(cfg_raw)
    if args.judge_model:
        jcfg.judge_models = tuple(args.judge_model)

    # Sample count: --smoke wins, then --n, then config default.
    if args.smoke:
        n_samples = jcfg.smoke_n
    elif args.n is not None:
        n_samples = args.n
    else:
        n_samples = jcfg.n_samples

    # 1. Load the three model groups' results.jsonl (M9 products).
    input_dirs = _resolve_input_dirs(args.inputs, cfg_raw.get("inputs"))
    if not input_dirs:
        logger.error("no inputs: pass --inputs or set 'inputs' in the config")
        return EXIT_CONTRACT
    try:
        results_by_tag = _load_results(input_dirs)
    except (FileNotFoundError, ValueError) as e:
        logger.error("input contract failure: %s", e)
        logger.error("Run M9 first: scripts/eval/run_offline_eval.py per group.")
        return EXIT_CONTRACT

    # 2. Select the SAME seeded, scenario-stratified batch for every group.
    try:
        samples_by_tag = select_judge_samples(results_by_tag, n_samples, jcfg.seed)
    except (ValueError, AssertionError, KeyError) as e:
        logger.error("sample selection failed: %s", e)
        return EXIT_CONTRACT
    n_per_group = len(next(iter(samples_by_tag.values())))
    sample_ids = sorted(r["id"] for r in next(iter(samples_by_tag.values())))
    logger.info(
        "scoring %d ids/group x %d groups x %d judge(s) = %d calls (tags=%s, judges=%s)",
        n_per_group, len(samples_by_tag), len(jcfg.judge_models),
        n_per_group * len(samples_by_tag) * len(jcfg.judge_models),
        list(samples_by_tag), list(jcfg.judge_models),
    )

    # 3. Judge via OpenRouter (blind). No key / total failure -> exit 3.
    template = load_template(cfg_raw["prompt_template_path"])
    load_env_file(find_repo_root() / ".env")
    try:
        client = _build_client(jcfg)
    except OpenRouterError as e:
        logger.error("cannot reach the judge API: %s", e)
        return EXIT_DEPENDENCY
    run = asyncio.run(run_judge(samples_by_tag, client, jcfg, template))
    if run.succeeded == 0 and run.attempted > 0:
        logger.error(
            "all %d judge calls failed (%s) -- check API key/quota and judge model ids",
            run.attempted, run.failures_by_kind,
        )
        return EXIT_DEPENDENCY

    # 4. Aggregate + cost.
    table = aggregate_scores(run.scores, jcfg.dimensions, jcfg.no_diff_threshold)
    cost = estimate_cost(run.tokens_by_model, jcfg.pricing)

    # 5. Write products + manifest under output_dir (tracked; M12 report material).
    out_dir = Path(args.output_dir or cfg_raw["output_dir"])
    write_jsonl(out_dir / "scores.jsonl", (s.to_row() for s in run.scores))
    md = render_comparison_md(
        table, run, jcfg, cost=cost, n_per_group=n_per_group, sample_ids=sample_ids
    )
    (out_dir / "comparison.md").write_text(md, encoding="utf-8", newline="\n")
    with open(out_dir / "aggregate.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
        f.write("\n")

    manifest = build_manifest(
        inputs=[d / "results.jsonl" for d in input_dirs],
        config=cfg_raw,
        stats={
            "judge_models": list(jcfg.judge_models),
            "n_per_group": n_per_group,
            "model_tags": list(samples_by_tag),
            "sample_ids": sample_ids,
            "attempted": run.attempted,
            "succeeded": run.succeeded,
            "parse_failures": run.parse_failures,
            "failures_by_kind": run.failures_by_kind,
            "validation_retries": run.validation_retries,
            "usage": run.usage.as_dict(),
            "cost": cost,
        },
    )
    write_manifest(out_dir, manifest)

    logger.info(
        "done: %d scores -> %s | est. cost $%.4f%s",
        run.succeeded, out_dir, cost["total_usd"],
        f" | parse failures {run.parse_failures}" if run.parse_failures else "",
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
