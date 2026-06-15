"""M2 thin CLI: synthesize sales dialogues / DPO preference pairs via OpenRouter.

Usage:
    uv run python scripts/data/synthesize.py --config configs/synthesize.yaml \
        --mode dialogues|preferences [--smoke] [--output-dir DIR]

Expands the task matrix, renders prompts (with few-shot seed examples), calls
the strong model concurrently, runs the generation-side quality gate, and writes
``synthetic_dialogues.jsonl`` / ``preference_pairs.jsonl`` + a cost report +
``manifest.json``. ``--smoke`` generates only a couple per scenario for a cheap
real-API connectivity check.

Exit codes: 0 success, 2 input contract failure (config / zero valid records),
3 external dependency failure (missing API key / API errors).
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
from sales_agent.common.io import write_jsonl
from sales_agent.common.manifest import build_manifest
from sales_agent.common.openrouter import OpenRouterClient, OpenRouterError
from sales_agent.data.synthesize import (
    SynthConfig,
    SynthResult,
    expand_task_matrix,
    load_seeds,
    load_template,
    run_synthesis,
)

logger = logging.getLogger("synthesize")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_EXTERNAL = 3

# Per-mode wiring: where the template, seeds, and output live in the config.
MODE_SPEC = {
    "dialogues": {
        "template_key": "dialogue_path",
        "seed_key": "dialogue_path",
        "seed_field": "scenario",
        "filename": "synthetic_dialogues.jsonl",
    },
    "preferences": {
        "template_key": "preference_path",
        "seed_key": "preference_path",
        "seed_field": "failure_mode",
        "filename": "preference_pairs.jsonl",
    },
}


def load_env_file(path: Path) -> None:
    """Minimal .env loader (no extra dependency): KEY=VALUE lines into os.environ.

    Existing environment variables win; quotes around values are stripped.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def estimate_cost_usd(usage, pricing: dict) -> float | None:
    """Estimate USD cost from token usage and configured per-1M-token prices."""
    inp, out = pricing.get("input_per_1m"), pricing.get("output_per_1m")
    if inp is None or out is None:
        return None
    return round(usage.prompt_tokens / 1e6 * inp + usage.completion_tokens / 1e6 * out, 4)


def build_cost_report(mode: str, cfg: SynthConfig, result: SynthResult) -> dict:
    counts_by_scenario: dict[str, int] = {}
    for rec in result.records:
        key = rec.get("scenario", "unknown")
        counts_by_scenario[key] = counts_by_scenario.get(key, 0) + 1
    return {
        "mode": mode,
        "model": cfg.model,
        "attempted": result.attempted,
        "succeeded": result.succeeded,
        "abandoned": result.abandoned,
        "validation_retries": result.validation_retries,
        "errors_by_kind": result.errors_by_kind,
        "records_by_scenario": dict(sorted(counts_by_scenario.items())),
        "usage": result.usage.as_dict(),
        "estimated_cost_usd": estimate_cost_usd(result.usage, cfg.pricing),
        "abandoned_samples": result.abandoned_samples,
    }


def main(argv: list[str] | None = None, *, client: OpenRouterClient | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_SPEC))
    parser.add_argument("--smoke", action="store_true", help="generate a couple per scenario")
    parser.add_argument("--output-dir", default=None, help="override output directory")
    parser.add_argument("--model", default=None, help="override config model id")
    parser.add_argument(
        "--per-scenario",
        type=int,
        default=None,
        help="cap generated records per scenario (e.g. to bootstrap a seed pool)",
    )
    args = parser.parse_args(argv)

    cfg_dict = load_config(args.config)
    if args.model:
        cfg_dict["model"] = args.model
    try:
        cfg = SynthConfig.from_dict(cfg_dict)
    except KeyError as e:
        logger.error("config missing required key: %s", e)
        return EXIT_CONTRACT

    spec = MODE_SPEC[args.mode]
    mode_cfg = cfg_dict.get(args.mode, {})
    templates = cfg_dict.get("templates", {})
    seeds_cfg = cfg_dict.get("seeds", {})

    template_path = templates.get(spec["template_key"])
    if not template_path or not mode_cfg.get("output_path"):
        logger.error(
            "config needs templates.%s and %s.output_path", spec["template_key"], args.mode
        )
        return EXIT_CONTRACT

    out_path = (
        Path(args.output_dir) / spec["filename"]
        if args.output_dir
        else Path(mode_cfg["output_path"])
    )
    output_dir = out_path.parent

    # Few-shot seeds are optional but recommended; absence is logged, not fatal.
    seeds_by_key = None
    seed_path = seeds_cfg.get(spec["seed_key"])
    if seed_path and Path(seed_path).exists():
        seeds_by_key = load_seeds(seed_path, spec["seed_field"])
    elif cfg.seed_examples_range[1] > 0:
        logger.warning("no seed file at %s; generating without few-shot examples", seed_path)

    if args.per_scenario is not None:
        per_scenario_limit = args.per_scenario
    elif args.smoke:
        per_scenario_limit = cfg.smoke_per_scenario
    else:
        per_scenario_limit = None
    try:
        tasks = expand_task_matrix(
            cfg, args.mode, seeds_by_key=seeds_by_key, per_scenario_limit=per_scenario_limit
        )
    except (KeyError, ValueError) as e:
        logger.error("input contract failure expanding task matrix: %s", e)
        return EXIT_CONTRACT
    if not tasks:
        logger.error("task matrix expanded to 0 tasks; check %s config section", args.mode)
        return EXIT_CONTRACT

    template = load_template(template_path)

    if client is None:
        load_env_file(find_repo_root() / ".env")
        try:
            client = OpenRouterClient(
                cfg.model, concurrency=cfg.concurrency, max_retries=cfg.client_max_retries
            )
        except OpenRouterError as e:
            logger.error("external dependency failure: %s", e)
            return EXIT_EXTERNAL

    logger.info(
        "synthesizing %d %s task(s) with %s (concurrency=%d, smoke=%s)",
        len(tasks),
        args.mode,
        cfg.model,
        cfg.concurrency,
        args.smoke,
    )
    result = asyncio.run(run_synthesis(tasks, client, cfg, template))

    report = build_cost_report(args.mode, cfg, result)
    report_path = output_dir / f"{args.mode}_cost_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not result.records:
        logger.error(
            "produced 0 valid records (errors=%s); see %s", result.errors_by_kind, report_path
        )
        return EXIT_EXTERNAL if result.errors_by_kind.get("api_error") else EXIT_CONTRACT

    n = write_jsonl(out_path, result.records)

    seed_inputs = [template_path] + ([seed_path] if seeds_by_key else [])
    manifest = build_manifest(
        inputs=seed_inputs,
        config=cfg_dict,
        stats={k: report[k] for k in ("attempted", "succeeded", "abandoned", "estimated_cost_usd")},
    )
    # Per-mode manifest name: M1 and both M2 modes share data/interim/, so the
    # default fixed "manifest.json" would clobber each other.
    manifest_path = output_dir / f"{args.mode}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info(
        "wrote %d records -> %s | succeeded=%d abandoned=%d retries=%d | "
        "tokens=%d est_cost=$%s | report=%s",
        n,
        out_path,
        result.succeeded,
        result.abandoned,
        result.validation_retries,
        result.usage.total_tokens,
        report["estimated_cost_usd"],
        report_path,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
