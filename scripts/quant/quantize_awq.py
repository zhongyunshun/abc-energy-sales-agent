"""M7 thin CLI: AWQ INT4 (W4A16) PTQ of the merged model via llm-compressor.

Goal (design doc section 3-M7 / proposal section 4-C1): quantize the M6 merged
BF16 model to W4A16 (asymmetric, group_size 128) with an activation-aware AWQ
recipe, calibrated on text sampled from the training domain, producing a
``models/quantized/awq/`` directory that transformers (and M8/vLLM) can load.

Pipeline:
  1. validate the merged model + train.jsonl are present (exit 2 otherwise);
  2. build the calibration set (pure logic in sales_agent.quant.calibration):
     stratified-proportional sample of train.jsonl rendered to ChatML text;
  3. BF16-load the merged model on CPU; run llm-compressor ``oneshot`` with the
     AWQ recipe -- the `sequential` pipeline onloads ONE block at a time to the
     GPU, so peak GPU stays a few GB (4070-safe; see configs/quant.yaml);
  4. save the compressed model + tokenizer + chat template to output_dir;
  5. self-check: reload via transformers + a single greedy generation (T7.3);
  6. size accounting (FP16 vs INT4) + manifest; commit evidence under reports/
     (manifest copy + size/probe report) on full runs;
  7. optional 5-prompt FP16-vs-INT4 probe (T7.4) for the completion report.

Heavy GPU deps (torch/transformers/llmcompressor/datasets) are imported lazily
inside main() so this file imports cleanly on the CPU-only Windows host. Exit
codes (design doc 1.4): 0 success, 2 input-contract failure (missing merged model
or train.jsonl), 3 external dependency failure (GPU/import/CUDA unavailable, or a
broken artifact that does not reload).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.quant.calibration import (
    load_calibration_texts,
    model_dir_size_bytes,
    size_report,
)
from sales_agent.training.formatting import render_chatml

logger = logging.getLogger("quantize_awq")

EXIT_OK = 0
EXIT_CONTRACT = 2  # missing merged model or calibration source
EXIT_DEPENDENCY = 3  # GPU/import unavailable, or saved artifact does not reload


def _drop_transformers_version(model_dir: Path) -> bool:
    """Drop the optional ``transformers_version`` field from config.json.

    Same transformers<=4.57.2 local-load workaround M6 applied to the merged dir
    (AutoTokenizer crashes on a local non-mistral model whose config declares that
    field). Harmless on fixed releases (the field is informational). Returns True
    if the field was removed.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if "transformers_version" not in cfg:
        return False
    cfg.pop("transformers_version")
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def _build_awq_recipe(rcfg: dict):
    """Construct the AWQ recipe modifier from config.

    Primary path is the documented single-modifier AWQ (llm-compressor's official
    examples): ``AWQModifier(scheme="W4A16_ASYM", targets=["Linear"],
    ignore=["lm_head"], duo_scaling=...)`` which performs the activation-aware
    scale search AND the W4A16 quantization. In llm-compressor 0.11 this import is a
    compatibility shim (still functional); if a future image drops it, fall back to
    the split transform+quantization API. group_size is fixed at 128 by the
    W4A16_ASYM preset, so it is asserted (not re-passed) by the caller.
    """
    kwargs = {
        "scheme": rcfg["scheme"],
        "targets": rcfg["targets"],
        "ignore": rcfg["ignore"],
    }
    if rcfg.get("duo_scaling") is not None:
        kwargs["duo_scaling"] = rcfg["duo_scaling"]
    try:
        from llmcompressor.modifiers.awq import AWQModifier

        return AWQModifier(**kwargs)
    except Exception as e:  # noqa: BLE001 -- shim removed: use the split API instead
        logger.info("single-modifier AWQ unavailable (%s); using transform+quant split API", e)
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.awq import AWQModifier as AWQTransformModifier

        transform = (
            AWQTransformModifier(duo_scaling=rcfg["duo_scaling"])
            if rcfg.get("duo_scaling") is not None
            else AWQTransformModifier()
        )
        return [
            transform,
            QuantizationModifier(
                scheme=rcfg["scheme"], targets=rcfg["targets"], ignore=rcfg["ignore"]
            ),
        ]


def _greedy(model, tokenizer, rendered: str, max_new_tokens: int, device: str) -> str:
    """Greedy-decode a single rendered ChatML prompt; return only the new text."""
    import torch

    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _run_probe(merged_dir: Path, output_dir: Path, pcfg: dict, dtype) -> str | None:
    """Generate the 5 probe prompts on FP16 (merged) and INT4 (quantized) models.

    Loads the two models sequentially (INT4 first, then frees it before the ~8GB
    FP16 load) so the 12GB card is never asked to hold both. Returns a Markdown
    side-by-side, or None if the fixture is missing. Best-effort: a failure here is
    logged but does not fail the quantization (the product is already saved).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts_path = Path(pcfg["prompts_path"])
    if not prompts_path.exists():
        logger.warning("probe prompts not found at %s; skipping FP16-vs-INT4 probe", prompts_path)
        return None
    probes = list(read_jsonl(prompts_path))
    max_new = pcfg["max_new_tokens"]
    # Render each probe context to a generation prompt (reuse render_chatml).
    from sales_agent.common.schema import Message

    rendered = [
        render_chatml([Message(**m) for m in p["context"]], add_generation_prompt=True)
        for p in probes
    ]

    def generate_all(model_dir: Path) -> list[str]:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=dtype, device_map="auto"
        )
        model.eval()
        dev = model.device if hasattr(model, "device") else "cuda"
        outs = [_greedy(model, tok, r, max_new, str(dev)) for r in rendered]
        del model
        torch.cuda.empty_cache()
        return outs

    int4_outs = generate_all(output_dir)
    fp16_outs = generate_all(merged_dir)

    lines = ["# M7 AWQ FP16-vs-INT4 probe (greedy)\n"]
    for p, fp, q in zip(probes, fp16_outs, int4_outs, strict=True):
        lines.append(f"## {p['id']}\n")
        lines.append(f"**FP16 (merged):** {fp.strip()}\n")
        lines.append(f"**INT4 (AWQ):** {q.strip()}\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--merged-dir", default=None, help="override config model.merged_dir")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    parser.add_argument("--smoke", action="store_true", help="tiny calibration (smoke_n_samples)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg, rcfg, calc = cfg["model"], cfg["recipe"], cfg["calibration"]
    merged_dir = Path(args.merged_dir or mcfg["merged_dir"])
    output_dir = Path(args.output_dir or cfg["output_dir"])
    n_calib = calc["smoke_n_samples"] if args.smoke else calc["n_samples"]
    max_seq_len = calc["max_seq_len"]
    seed = cfg["seed"]

    # group_size is fixed at 128 by the W4A16_ASYM preset; refuse a config that
    # silently disagrees rather than quantizing at an unexpected group size.
    if rcfg["scheme"] == "W4A16_ASYM" and rcfg.get("group_size", 128) != 128:
        logger.error(
            "recipe.group_size=%s but the W4A16_ASYM preset is fixed at 128; "
            "use config_groups for a different group size.",
            rcfg.get("group_size"),
        )
        return EXIT_CONTRACT

    # 1. Contracts: merged model + calibration source must exist.
    if not (merged_dir / "config.json").exists():
        logger.error(
            "merged model not found at %s -- M7 quantizes the M6 merged product. "
            "On this PC it lives under models/adapters/merged/ (see configs/quant.yaml).",
            merged_dir,
        )
        return EXIT_CONTRACT
    train_path = Path(calc["source_path"])
    if not train_path.exists():
        logger.error("calibration source not found at %s (M3 train.jsonl)", train_path)
        return EXIT_CONTRACT

    # 2. Build calibration texts (pure logic, before any heavy import).
    texts, calib_report = load_calibration_texts(train_path, n_calib, seed)
    logger.info(
        "calibration: %d texts (requested %d) | per-scenario=%s | max_seq_len=%d",
        calib_report.n_selected, n_calib, calib_report.per_scenario, max_seq_len,
    )

    # 3. Lazy GPU imports (exit 3 if the stack is unavailable).
    try:
        import torch
        from datasets import Dataset
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error("GPU quant stack unavailable (run inside the train container): %s", e)
        return EXIT_DEPENDENCY
    if not torch.cuda.is_available():
        logger.error(
            "CUDA not available -- AWQ calibration needs the GPU even with CPU offload. "
            "Run inside the train container on the 4070."
        )
        return EXIT_DEPENDENCY

    dtype = getattr(torch, mcfg["dtype"])  # e.g. torch.bfloat16

    # 4. Load tokenizer + merged model (BF16, on CPU; oneshot onloads blocks to GPU).
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(
        "loading merged model %s (dtype=%s, device_map=%s)",
        merged_dir, mcfg["dtype"], mcfg["device_map"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), dtype=dtype, device_map=mcfg["device_map"]
    )

    # 5. AWQ oneshot. The calibration Dataset is a single "text" column; oneshot
    #    tokenizes + truncates to max_seq_length and runs the sequential pipeline.
    calib_ds = Dataset.from_dict({"text": texts})
    recipe = _build_awq_recipe(rcfg)
    logger.info(
        "running AWQ oneshot: scheme=%s group_size=%d samples=%d pipeline=%s batch=%d",
        rcfg["scheme"], rcfg.get("group_size", 128), len(texts),
        calc["pipeline"], calc["batch_size"],
    )
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        num_calibration_samples=len(texts),
        max_seq_length=max_seq_len,
        batch_size=calc["batch_size"],
        text_column="text",
        pipeline=calc["pipeline"],
    )

    # 6. Save the compressed model + tokenizer + chat template (the M8 contract).
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(output_dir))
    patched = _drop_transformers_version(output_dir)
    if patched:
        logger.info("dropped config.json transformers_version (transformers<=4.57.2 load fix)")

    # 7. Self-check (T7.3): reload from disk + a single greedy generation.
    self_check_ok = False
    self_check_text = ""
    try:
        del model
        torch.cuda.empty_cache()
        scfg = cfg["self_check"]
        tok2 = AutoTokenizer.from_pretrained(str(output_dir))
        if tok2.pad_token is None:
            tok2.pad_token = tok2.eos_token
        int4 = AutoModelForCausalLM.from_pretrained(str(output_dir), device_map="auto")
        from sales_agent.common.schema import Message

        rendered = render_chatml(
            [Message(role="user", content=scfg["prompt"])], add_generation_prompt=True
        )
        dev = str(int4.device) if hasattr(int4, "device") else "cuda"
        self_check_text = _greedy(int4, tok2, rendered, scfg["max_new_tokens"], dev)
        self_check_ok = len(self_check_text.strip()) > 0
        logger.info("self-check generation OK: %r", self_check_text[:120])
        del int4
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 -- a broken artifact must surface, not be masked
        logger.error("self-check FAILED: saved INT4 model did not reload/generate: %s", e)

    # 8. Size accounting + manifest.
    fp16_bytes = model_dir_size_bytes(merged_dir)
    int4_bytes = model_dir_size_bytes(output_dir)
    sizes = size_report(fp16_bytes, int4_bytes)
    logger.info(
        "sizes: FP16 %.3f GB -> INT4 %.3f GB (%.2f%% smaller, %.2fx)",
        sizes["fp16_gb"], sizes["int4_gb"], sizes["size_reduction_pct"] or 0.0,
        sizes["compression_ratio"] or 0.0,
    )
    stats = {
        "merged_dir": str(merged_dir),
        "smoke": args.smoke,
        "recipe": {
            "scheme": rcfg["scheme"],
            "group_size": rcfg.get("group_size", 128),
            "symmetric": rcfg.get("symmetric", False),
            "targets": rcfg["targets"],
            "ignore": rcfg["ignore"],
            "duo_scaling": rcfg.get("duo_scaling"),
        },
        "calibration": calib_report.as_dict(),
        "max_seq_len": max_seq_len,
        "device_map": mcfg["device_map"],
        "pipeline": calc["pipeline"],
        "sizes": sizes,
        "config_transformers_version_dropped": patched,
        "self_check_ok": self_check_ok,
        "self_check_sample": self_check_text[:200],
    }
    inputs = []
    for shard in sorted(merged_dir.glob("*.safetensors")):
        inputs.append(shard)
    if train_path.exists():
        inputs.append(train_path)
    manifest = build_manifest(inputs=inputs, config=cfg, stats=stats)
    write_manifest(output_dir, manifest)

    # 9. Optional FP16-vs-INT4 probe (T7.4), full runs only.
    probe_md = None
    pcfg = cfg.get("probe", {})
    if pcfg.get("enabled") and not args.smoke:
        try:
            probe_md = _run_probe(merged_dir, output_dir, pcfg, dtype)
        except Exception as e:  # noqa: BLE001 -- probe is evidence, not a gate
            logger.error("FP16-vs-INT4 probe failed (non-fatal): %s", e)

    # 10. Committed evidence under reports/training/ (full runs only; the model dirs
    #     are gitignored). Smoke writes ONLY the manifest into output_dir.
    if not args.smoke:
        rep = cfg["report"]
        rep_manifest = Path(rep["manifest_path"])
        rep_size = Path(rep["size_report_path"])
        rep_manifest.parent.mkdir(parents=True, exist_ok=True)
        rep_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        size_md = _render_size_report(stats, manifest)
        if probe_md:
            size_md = size_md + "\n\n" + probe_md
        rep_size.write_text(size_md, encoding="utf-8", newline="\n")
        logger.info("committed evidence -> %s , %s", rep_manifest, rep_size)

    if not self_check_ok:
        logger.error(
            "AWQ product saved to %s but the self-check generation failed -- the "
            "artifact may not be loadable (DoD). Exiting 3.",
            output_dir,
        )
        return EXIT_DEPENDENCY

    logger.info(
        "quant done -> %s | FP16 %.2f GB -> INT4 %.2f GB | self-check PASS",
        output_dir, sizes["fp16_gb"], sizes["int4_gb"],
    )
    return EXIT_OK


def _render_size_report(stats: dict, manifest: dict) -> str:
    """Human-readable FP16-vs-INT4 size report (committed under reports/training/)."""
    s = stats["sizes"]
    r = stats["recipe"]
    cov = stats["calibration"].get("per_scenario", {})
    return (
        "# M7 AWQ quantization report\n\n"
        f"- git_commit: `{manifest.get('git_commit')}`\n"
        f"- created_at: {manifest.get('created_at')}\n"
        f"- merged (FP16) dir: `{stats['merged_dir']}`\n"
        f"- recipe: {r['scheme']} (group_size {r['group_size']}, "
        f"symmetric={r['symmetric']}, duo_scaling={r['duo_scaling']})\n"
        f"- calibration: {stats['calibration']['n_selected']} samples, "
        f"max_seq_len {stats['max_seq_len']}, per-scenario {cov}\n"
        f"- device_map: {stats['device_map']}, pipeline: {stats['pipeline']}\n\n"
        "## Size trade-off (README source)\n\n"
        "| | bytes | GB |\n|---|---:|---:|\n"
        f"| FP16 (merged) | {s['fp16_bytes']:,} | {s['fp16_gb']} |\n"
        f"| INT4 (AWQ) | {s['int4_bytes']:,} | {s['int4_gb']} |\n\n"
        f"- compression ratio: **{s['compression_ratio']}x**\n"
        f"- size reduction: **{s['size_reduction_pct']}%**\n"
        f"- self-check generation: {'PASS' if stats['self_check_ok'] else 'FAIL'} "
        f"(`{stats['self_check_sample']}`)\n"
    )


if __name__ == "__main__":
    sys.exit(main())
