"""M6 thin CLI: merge the DPO LoRA adapter into dense BF16 weights + consistency check.

Goal (the M6 merge target / the M6 contract): fold the M5 DPO adapter
into the base model so M7 (AWQ quant) and M8 (vLLM) can load a plain HF model
directory, and prove the fold is behaviour-preserving with an 8-prompt greedy
consistency check.

Pipeline (the M6 contract):
  1. validate the adapter is in place (exit 2 if missing -- never merge nothing);
  2. BF16-load base + the DPO adapter via PEFT (4-bit merging would lose precision);
  3. greedy-generate the 8 fixed prompts on `base + adapter` (PEFT inference);
  4. `merge_and_unload` -> dense BF16 model (in place: one model resident, so a
     12GB card holds only ~9GB at a time, not two copies);
  5. greedy-generate the same prompts on the merged model;
  6. compare prompt by prompt (exact, or config-relaxed prefix_tokens/N);
  7. save safetensors + tokenizer + chat template + manifest (consistency verdict
     ALWAYS recorded); exit 2 + print diffs if inconsistent.

Memory-efficient ordering note: step 3 runs BEFORE the merge and step 5 AFTER, on
the same single model object, so peak memory is one bf16 model -- important for the
12GB target card and the `device_map: cpu` fallback.

Heavy GPU deps (torch/transformers/peft) are imported lazily inside main() so this
file imports cleanly on a CPU-only host. Exit codes: 0 success, 2 input-contract
failure (missing adapter) OR consistency mismatch, 3 external dependency failure
(GPU/import/CUDA unavailable).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.training.merge_consistency import (
    Generation,
    compare_generations,
    load_consistency_prompts,
    render_consistency_report,
)

logger = logging.getLogger("merge_adapter")

EXIT_OK = 0
EXIT_CONTRACT = 2  # missing adapter OR consistency mismatch
EXIT_DEPENDENCY = 3


def _normalize_config_for_local_load(output_dir: Path) -> bool:
    """Drop the optional ``transformers_version`` field from the merged config.json.

    Works around a transformers 4.57.2 bug: ``AutoTokenizer.from_pretrained`` on a
    *local* directory runs a mistral-regex-fix detection that does
    ``_config = json.load(config.json)`` (a plain dict) and then accesses
    ``_config.model_type`` (attribute, not ``.get()``) -- crashing with
    ``AttributeError: 'dict' object has no attribute 'model_type'`` for any local
    non-mistral model whose config declares ``transformers_version <= 4.57.2``. The
    buggy branch is gated on that field being present, so removing this purely
    informational metadata makes the merged dir loadable. The model is unaffected
    (it does not read ``transformers_version``). Returns True if the field was
    removed. Forward-compatible: the field is optional, so dropping it is harmless
    on transformers releases where the bug is fixed.
    """
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return False
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if "transformers_version" not in cfg:
        return False
    cfg.pop("transformers_version")
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="override config adapter dir")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    parser.add_argument("--smoke", action="store_true", help="tiny run (few prompts, short gen)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg, ccfg = cfg["model"], cfg["consistency"]
    adapter_dir = Path(args.adapter or cfg["adapter"])
    output_dir = Path(args.output_dir or cfg["output_dir"])
    match_mode = ccfg["match_mode"]
    prefix_n = ccfg["prefix_n"]
    max_new = cfg["smoke"]["max_new_tokens"] if args.smoke else ccfg["max_new_tokens"]

    # 1. Adapter must be in place -- M6 merges a real adapter, never falls back to base.
    if not (adapter_dir / "adapter_config.json").exists():
        logger.error(
            "adapter not found at %s -- M6 merges the M5 DPO adapter (the standalone "
            "policy copy at the adapter dir root, NOT the frozen reference). Transfer "
            "the adapter to this machine before merging.",
            adapter_dir,
        )
        return EXIT_CONTRACT

    # 2. Fixed consistency prompts (loud failure on a malformed fixture).
    prompts = load_consistency_prompts(
        str(Path(ccfg["prompts_path"])), default_system=ccfg.get("default_system")
    )
    if args.smoke:
        prompts = prompts[: cfg["smoke"]["n_prompts"]]
    logger.info(
        "merging adapter=%s -> %s | %d consistency prompts | mode=%s | smoke=%s",
        adapter_dir, output_dir, len(prompts), match_mode, args.smoke,
    )

    # 3. Lazy GPU imports (exit 3 if the stack is unavailable).
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error("GPU merge stack unavailable (run inside the train env): %s", e)
        return EXIT_DEPENDENCY

    device_map = mcfg["device_map"]
    on_cpu = device_map == "cpu"
    if not on_cpu and not torch.cuda.is_available():
        logger.error(
            "CUDA not available and device_map=%s -- set model.device_map: cpu to "
            "merge on CPU (slow but stable), or run on the GPU node.",
            device_map,
        )
        return EXIT_DEPENDENCY
    gen_device = "cpu" if on_cpu else "cuda"

    dtype = getattr(torch, mcfg["dtype"])  # e.g. torch.bfloat16

    # 4. Tokenizer + BF16 base + DPO adapter (PEFT inference model).
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["name"], dtype=dtype, device_map=device_map
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    peft_model.eval()

    def generate(model) -> list[Generation]:
        """Greedy-generate every fixed prompt, capturing decoded text + new token ids."""
        model.eval()
        model.config.use_cache = True
        gens: list[Generation] = []
        for p in prompts:
            inputs = tokenizer(p.rendered, return_tensors="pt").to(gen_device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            gens.append(Generation(prompt_id=p.id, text=text, token_ids=new_ids.tolist()))
        return gens

    # 5. BEFORE merge: generate on base + adapter (PEFT inference).
    logger.info("generating %d prompts on base+adapter (PEFT)...", len(prompts))
    peft_gens = generate(peft_model)

    # 6. Merge the adapter into dense weights (in place -> one model resident).
    logger.info("merging adapter into dense %s weights (merge_and_unload)...", mcfg["dtype"])
    merged_model = peft_model.merge_and_unload()

    # 7. AFTER merge: generate the same prompts on the merged model.
    logger.info("generating %d prompts on merged model...", len(prompts))
    merged_gens = generate(merged_model)

    # 8. Consistency check.
    result = compare_generations(peft_gens, merged_gens, mode=match_mode, prefix_n=prefix_n)
    logger.info(
        "consistency: %s (%d/%d prompts match, mode=%s)",
        "PASS" if result.consistent else "FAIL",
        result.n_total - result.n_mismatch, result.n_total, match_mode,
    )

    # 9. Save the merged model + tokenizer + chat template (the M7/M8 contract).
    #    Done regardless of the verdict so a failed run leaves an inspectable artifact;
    #    the manifest records the verdict and the exit code gates the pipeline.
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))  # tokenizer + chat_template.jinja

    # 9b. Make the merged dir directly loadable (transformers 4.57.2 tokenizer bug)
    #     and self-verify the DoD "transformers can load it" by reloading the tokenizer.
    patched = _normalize_config_for_local_load(output_dir)
    if patched:
        logger.info("dropped config.json transformers_version (transformers<=4.57.2 load fix)")
    try:
        from transformers import AutoTokenizer as _AutoTok

        _AutoTok.from_pretrained(str(output_dir))
        tokenizer_reload_ok = True
        logger.info("self-check: AutoTokenizer.from_pretrained(merged) OK")
    except Exception as e:  # noqa: BLE001 -- surface a broken artifact, don't mask it
        tokenizer_reload_ok = False
        logger.error("self-check FAILED: merged dir tokenizer does not reload: %s", e)

    # 10. Manifest: provenance + consistency verdict (DoD: recorded in manifest).
    weights = output_dir / "model.safetensors"
    sharded = list(output_dir.glob("model-*.safetensors"))
    merged_bytes = (
        weights.stat().st_size if weights.exists()
        else sum(p.stat().st_size for p in sharded)
    )
    stats = {
        "adapter": str(adapter_dir),
        "base_model": mcfg["name"],
        "merge_dtype": mcfg["dtype"],
        "device_map": device_map,
        "merged_on_cpu": on_cpu,
        "smoke": args.smoke,
        "max_new_tokens": max_new,
        "merged_size_gb": round(merged_bytes / 1e9, 3),
        "config_transformers_version_dropped": patched,
        "tokenizer_reload_ok": tokenizer_reload_ok,
        "consistency_check": result.summary(),
    }
    inputs = []
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if adapter_weights.exists():
        inputs.append(adapter_weights)
    manifest = build_manifest(inputs=inputs, config=cfg, stats=stats)
    write_manifest(output_dir, manifest)

    # 10b. Committable evidence: a human-readable consistency report (and, for full
    #      runs, a manifest copy) under reports/training/ so the merge + 8/8 verdict
    #      are auditable from the repo (the merged model dir is gitignored). The
    #      report always travels with the model in output_dir too. Smoke runs write
    #      ONLY into output_dir so they never clobber the committed evidence.
    report_meta = {
        "base": mcfg["name"],
        "adapter": str(adapter_dir),
        "dtype": mcfg["dtype"],
        "device_map": device_map,
        "merged_size_gb": stats["merged_size_gb"],
        "git_commit": manifest.get("git_commit"),
        "created_at": manifest.get("created_at"),
    }
    report_md = render_consistency_report(result, merged_gens, meta=report_meta)
    (output_dir / "consistency_report.md").write_text(report_md, encoding="utf-8", newline="\n")
    if not args.smoke:
        rep_md = Path(cfg["report"]["consistency_report_path"])
        rep_manifest = Path(cfg["report"]["manifest_path"])
        rep_md.parent.mkdir(parents=True, exist_ok=True)
        rep_md.write_text(report_md, encoding="utf-8", newline="\n")
        rep_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("committed evidence -> %s , %s", rep_md, rep_manifest)

    if not result.consistent:
        logger.error(
            "MERGE CONSISTENCY FAILED -- merged behaviour differs from base+adapter.\n%s",
            result.render_diffs(),
        )
        logger.error(
            "merged model + manifest saved to %s for inspection; exiting 2. If the "
            "diffs are only benign late-token bf16 drift, relax consistency.match_mode "
            "to 'prefix_tokens' in the config and re-run.",
            output_dir,
        )
        return EXIT_CONTRACT

    if not tokenizer_reload_ok:
        logger.error(
            "merged model + manifest saved to %s but the tokenizer does not reload "
            "via transformers -- the artifact is not directly loadable (DoD). Exiting 3.",
            output_dir,
        )
        return EXIT_DEPENDENCY

    logger.info(
        "merge done -> %s | %.2f GB | consistency PASS (%d/%d, mode=%s)",
        output_dir, stats["merged_size_gb"], result.n_total, result.n_total, match_mode,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
