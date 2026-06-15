"""M1 thin CLI: normalize raw sales dialogues into DialogueRecord JSONL.

Usage:
    uv run python scripts/data/normalize.py --config configs/normalize.yaml \
        [--smoke] [--output-dir DIR]

Loads each configured source (local json/jsonl or HF datasets), runs the
normalize pipeline (convert -> clean -> dedup -> tag -> validate), and writes
``normalized.jsonl`` + ``normalize_report.json`` + ``manifest.json`` into the
output directory. Exit codes: 0 success, 2 input contract failure, 3 external
dependency failure (HF download).
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
from sales_agent.data.normalize import KNOWN_FORMATS, CleanConfig, SourceBatch, run_pipeline

logger = logging.getLogger("normalize")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_EXTERNAL = 3


def load_local_source(path: Path) -> list[dict]:
    """Load a local .jsonl (one object per line) or .json (array) file."""
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array in {path}, got {type(data).__name__}")
    return data


def load_hf_source(dataset: str, split: str) -> list[dict]:
    """Download a HuggingFace dataset split and return its rows as dicts."""
    from datasets import load_dataset  # deferred: needs network, not used in unit tests

    ds = load_dataset(dataset, split=split)
    return [dict(row) for row in ds]


def load_sources(cfg: dict, limit: int | None) -> list[SourceBatch]:
    batches: list[SourceBatch] = []
    for spec in cfg.get("sources", []):
        tag, fmt = spec.get("source_tag"), spec.get("format")
        if not tag or fmt not in KNOWN_FORMATS:
            raise ValueError(
                f"source needs source_tag and format in {'|'.join(KNOWN_FORMATS)}: {spec}"
            )
        if "path" in spec:
            records = load_local_source(Path(spec["path"]))
        elif "hf_dataset" in spec:
            records = load_hf_source(spec["hf_dataset"], spec.get("split", "train"))
        else:
            raise ValueError(f"source {tag!r} needs either path or hf_dataset")
        if limit is not None:
            records = records[:limit]
        logger.info("loaded %d records from %s", len(records), tag)
        batches.append(SourceBatch(source_tag=tag, format=fmt, records=records))
    return batches


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="cap records per source")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["output_dir"])
    if not cfg.get("sources"):
        logger.error("config has no sources")
        return EXIT_CONTRACT
    limit = cfg.get("smoke_limit", 50) if args.smoke else None

    try:
        batches = load_sources(cfg, limit)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        logger.error("input contract failure: %s", e)
        return EXIT_CONTRACT
    except Exception as e:  # HF hub/network errors
        logger.error("external dependency failure loading HF dataset: %s", e)
        return EXIT_EXTERNAL

    rules = CleanConfig(**cfg.get("clean", {}))
    records, report = run_pipeline(batches, rules, cfg.get("scenario_keywords", {}))
    if not records:
        logger.error("pipeline produced 0 records; report: %s", json.dumps(report))
        return EXIT_CONTRACT

    out_path = output_dir / "normalized.jsonl"
    n = write_jsonl(out_path, (r.model_dump() for r in records))
    report_path = output_dir / "normalize_report.json"
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    local_inputs = [s["path"] for s in cfg["sources"] if "path" in s]
    manifest = build_manifest(inputs=local_inputs, config=cfg, stats=report["totals"])
    write_manifest(output_dir, manifest)

    logger.info(
        "wrote %d records -> %s | in=%d out=%d dropped=%s | report=%s",
        n,
        out_path,
        report["totals"]["input"],
        report["totals"]["output"],
        report["totals"]["dropped"],
        report_path,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
