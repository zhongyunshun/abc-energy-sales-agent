"""M9 thin CLI: generate replies for the test set and score behavioral rules.

Usage (endpoint path, after `bash scripts/serving/serve.sh`):
    uv run python scripts/eval/run_offline_eval.py --config configs/eval_offline.yaml \
        --model-tag dpo [--endpoint http://127.0.0.1:8000/v1] [--smoke]

Usage (local fallback, no server -- base or un-deployed adapter):
    uv run python scripts/eval/run_offline_eval.py --config configs/eval_offline.yaml \
        --model-tag base --local-model <hf-path-or-id> [--local-adapter <peft-dir>]

Flow (the M9 contract): build_eval_samples(test.jsonl) -> select_samples(seeded) ->
generate (endpoint async OR local transformers) -> strip_reasoning -> apply_rules ->
write reports/eval_offline/<model_tag>/{results.jsonl, summary.json, manifest.json}.

All three groups run with the SAME config so they see the identical sample batch and
generation params (DoD: same caliber). Exit codes (the CLI contract): 0 success, 2 input
contract failure (no test set / malformed records / no samples), 3 external dependency
failure (endpoint unreachable / local stack missing).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl, write_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.evals.generate import GenConfig, generate_all
from sales_agent.evals.rules import RuleConfig, apply_rules
from sales_agent.evals.samples import build_eval_samples, select_samples, strip_reasoning
from sales_agent.evals.summary import summarize_results

logger = logging.getLogger("run_offline_eval")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_DEPENDENCY = 3


def _build_rows(samples, outputs, rule_cfg, gen_cfg, model_tag):
    """Strip -> score every (sample, output) into M9 result rows (the result contract)."""
    gen_record = {**gen_cfg.as_record(), "model_tag": model_tag}
    rows: list[dict] = []
    for sample, out in zip(samples, outputs, strict=True):
        completion = strip_reasoning(out.content)
        flags, n_tokens = apply_rules(completion, sample.scenario, rule_cfg)
        rows.append(
            {
                "id": sample.id,
                "scenario": sample.scenario,
                "prompt_messages": sample.prompt_messages,
                "completion": completion,
                "rule_flags": flags,
                "n_tokens": n_tokens,
                "usage_completion_tokens": out.usage_completion_tokens,
                "gen_config": gen_record,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-tag", required=True, help="base | sft | dpo (output subdir)")
    parser.add_argument("--endpoint", default=None, help="OpenAI base URL (…/v1)")
    parser.add_argument("--served-model", default=None, help="override served model name")
    parser.add_argument("--local-model", default=None, help="HF model path/id for local fallback")
    parser.add_argument("--local-adapter", default=None, help="PEFT adapter dir for local fallback")
    parser.add_argument("--smoke", action="store_true", help="use smoke.n_samples (small run)")
    parser.add_argument("--n", type=int, default=None, help="override sampling.n_samples")
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg["seed"]
    gen_cfg = GenConfig.from_dict(cfg.get("generation"))
    rule_cfg = RuleConfig.from_dict(cfg.get("rules"))

    # 1. Load + build eval samples from the test set.
    test_path = Path(args.test_path or cfg["test_path"])
    if not test_path.exists():
        logger.error("test set not found at %s (M3 product)", test_path)
        return EXIT_CONTRACT
    try:
        samples = build_eval_samples(read_jsonl(test_path))
    except (ValueError, KeyError) as e:
        logger.error("malformed test record: %s", e)
        return EXIT_CONTRACT
    if not samples:
        logger.error("no eval samples built from %s", test_path)
        return EXIT_CONTRACT

    # 2. Select the (seeded, scenario-stratified) batch -- identical across groups.
    if args.smoke:
        n_samples = cfg.get("smoke", {}).get("n_samples", 10)
    elif args.n is not None:
        n_samples = args.n
    else:
        n_samples = cfg.get("sampling", {}).get("n_samples")
    batch = select_samples(samples, n_samples, seed)
    logger.info(
        "evaluating %d / %d samples (model-tag=%s)", len(batch), len(samples), args.model_tag
    )

    # 3. Generate replies -- endpoint (preferred) or local transformers fallback.
    if args.local_model or args.local_adapter:
        try:
            from sales_agent.evals.local_infer import local_generate
        except ImportError as e:
            logger.error("local inference stack unavailable: %s", e)
            return EXIT_DEPENDENCY
        if not args.local_model:
            logger.error("--local-adapter requires --local-model (the base to load it on)")
            return EXIT_CONTRACT
        try:
            outputs = local_generate(
                batch, gen_cfg, model_path=args.local_model, adapter=args.local_adapter
            )
        except Exception as e:  # noqa: BLE001 -- surface GPU/stack failure as exit 3
            logger.error("local generation failed: %s", e)
            return EXIT_DEPENDENCY
    else:
        endpoint = args.endpoint or cfg.get("endpoint", {}).get("default")
        model = args.served_model or cfg.get("endpoint", {}).get("served_model")
        if not endpoint or not model:
            logger.error("no endpoint/served_model configured (and no --local-model)")
            return EXIT_CONTRACT
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY")
            outputs = asyncio.run(generate_all(batch, client, model, gen_cfg))
        except Exception as e:  # noqa: BLE001 -- endpoint unreachable -> exit 3
            logger.error("generation failed against %s: %s", endpoint, e)
            logger.error("Is the server up? Run `bash scripts/serving/serve.sh` first.")
            return EXIT_DEPENDENCY

    # 4. Score + aggregate.
    rows = _build_rows(batch, outputs, rule_cfg, gen_cfg, args.model_tag)
    summary = summarize_results(rows, model_tag=args.model_tag, gen_config=gen_cfg.as_record())

    # 5. Write products + manifest under <output_dir>/<model_tag>/.
    out_dir = Path(args.output_dir or cfg["output_dir"]) / args.model_tag
    write_jsonl(out_dir / "results.jsonl", rows)
    import json

    with open(out_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    manifest = build_manifest(
        inputs=[test_path],
        config=cfg,
        stats={
            "model_tag": args.model_tag,
            "n_samples": len(rows),
            "overall_rule_rates": summary["overall"]["rule_rates"],
            "overall_length_tokens": summary["overall"]["length_tokens"],
        },
    )
    write_manifest(out_dir, manifest)

    o = summary["overall"]
    logger.info(
        "done: %d samples -> %s | rule_rates=%s | length p50/p95=%s/%s",
        len(rows), out_dir, o["rule_rates"],
        o["length_tokens"].get("p50"), o["length_tokens"].get("p95"),
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
