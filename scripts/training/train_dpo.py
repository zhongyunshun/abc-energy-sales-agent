"""M5 thin CLI: TRL DPO alignment on top of the M4 SFT adapter.

Goal (the M5 DPO target / the M5 contract): continue-train the SFT
adapter on synthetic preference pairs to suppress pushy closes and rate
hallucination.

Reference-model scheme (user-confirmed): "adapter 双副本" / ref=SFT. The base is
loaded ONCE in bf16; the SFT adapter is loaded twice on it -- a trainable `policy`
copy (named "default" so it saves to output_dir root) and a frozen `reference`
copy. TRL's DPOTrainer swaps to `ref_adapter_name` for reference logps, so KL is
anchored to SFT with no extra full model in memory.

Pipeline (the M5 contract):
  1. load + validate preference pairs (M2 product) and behaviour probes;
  2. bf16 base + two SFT-initialised adapters (policy trainable, reference frozen);
  3. PreferencePair -> DPO standard rows via formatting.preference_pair_to_dpo;
  4. greedy-generate the probes BEFORE training (= SFT behaviour);
  5. DPOTrainer.train (beta/lr/epochs from config); save the policy adapter;
  6. greedy-generate the probes AFTER training (= DPO behaviour) -> behaviour diff;
  7. loss + reward-margin curves + manifest (with peak VRAM, final margins).

Heavy GPU deps (torch/transformers/peft/trl) are imported lazily inside main() so
this file imports cleanly on a CPU-only host. Exit codes: 0 success, 2
input-contract failure (missing inputs / SFT adapter), 3 external dependency
failure (GPU/import/CUDA unavailable).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.common.schema import PreferencePair
from sales_agent.training.dpo_metrics import extract_reward_metrics, plot_reward_margins
from sales_agent.training.dpo_probes import (
    build_probe_prompt,
    count_changed,
    load_probes,
    render_behavior_diff,
)
from sales_agent.training.formatting import preference_pair_to_dpo
from sales_agent.training.plotting import plot_loss_curve

logger = logging.getLogger("train_dpo")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_DEPENDENCY = 3


def load_pairs(path: Path) -> tuple[list[PreferencePair], list[str]]:
    """Parse a JSONL file into validated PreferencePairs; collect error strings.

    The PreferencePair schema enforces the contract (context ends with user,
    non-empty responses, system only at index 0), so a structural failure here is
    a contract violation.
    """
    pairs: list[PreferencePair] = []
    errors: list[str] = []
    for i, raw in enumerate(read_jsonl(path)):
        try:
            pairs.append(PreferencePair.model_validate(raw))
        except Exception as e:  # pydantic structural/semantic failure
            errors.append(f"{path.name}:{i}: {e}")
    return pairs, errors


def _normalize_adapter_dir(output_dir: Path, policy_name: str) -> None:
    """Ensure the saved policy adapter lives at ``output_dir`` root.

    PEFT saves the "default" adapter to the root, but if a build saved it under
    ``output_dir/<policy_name>/`` instead, hoist its files up so the directory is a
    standalone, M6-mergeable adapter regardless of PEFT version.
    """
    nested = output_dir / policy_name
    root_has = (output_dir / "adapter_config.json").exists()
    if (nested / "adapter_config.json").exists() and not root_has:
        for f in nested.iterdir():
            shutil.move(str(f), str(output_dir / f.name))
        nested.rmdir()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny run (max_steps, n_pairs)")
    parser.add_argument("--output-dir", default=None, help="override config dpo.output_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg, dcfg, refcfg = cfg["model"], cfg["dpo"], cfg["ref"]
    datacfg, probecfg = cfg["data"], cfg["probes"]
    output_dir = Path(args.output_dir or dcfg["output_dir"])
    seed = cfg["seed"]
    policy_name, ref_name = refcfg["policy_adapter_name"], refcfg["ref_adapter_name"]
    default_system = datacfg.get("default_system")

    # 1. Load + validate inputs (exit 2 on contract failure).
    pref_path = Path(datacfg["pref_path"])
    if not pref_path.exists():
        logger.error("missing preference pairs %s (run M2 or provide a fixture)", pref_path)
        return EXIT_CONTRACT
    pairs, errors = load_pairs(pref_path)
    if errors:
        for e in errors[:10]:
            logger.error("invalid pair: %s", e)
        logger.error("%s: %d invalid pairs (contract failure)", pref_path.name, len(errors))
        return EXIT_CONTRACT

    sft_adapter = Path(cfg["sft_adapter"])
    if not (sft_adapter / "adapter_config.json").exists():
        logger.error(
            "SFT adapter not found at %s -- M5 continues training from the M4 SFT "
            "adapter and must NOT silently fall back to base. Transfer the adapter "
            "to this machine before running DPO.",
            sft_adapter,
        )
        return EXIT_CONTRACT

    probes = load_probes(probecfg["path"])
    if args.smoke:
        pairs = pairs[: cfg["smoke"]["n_pairs"]]
        probes = probes[: probecfg.get("smoke_n", 2)]
    logger.info(
        "loaded pairs=%d probes=%d sft_adapter=%s (smoke=%s)",
        len(pairs), len(probes), sft_adapter, args.smoke,
    )

    # 2. Lazy GPU imports (exit 3 if the stack/GPU is unavailable).
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        logger.error("GPU DPO stack unavailable (run inside the train env): %s", e)
        return EXIT_DEPENDENCY
    if not torch.cuda.is_available():
        logger.error("CUDA not available -- M5 must run on the GPU node")
        return EXIT_DEPENDENCY
    device = "cuda"

    # 3. Tokenizer + bf16 base + two SFT-initialised adapters.
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(mcfg["name"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(
        base, str(sft_adapter), adapter_name=policy_name, is_trainable=True
    )
    model.load_adapter(str(sft_adapter), adapter_name=ref_name)  # frozen reference
    model.set_adapter(policy_name)
    model.to(device)

    # 4. Build the DPO dataset (standard prompt/chosen/rejected rows).
    rows = [preference_pair_to_dpo(p, default_system=default_system) for p in pairs]
    dpo_ds = Dataset.from_list(rows)

    # 5. Greedy probe generation helper (shared by the before/after passes).
    def generate(prompts: list[str], max_new_tokens: int) -> list[str]:
        model.eval()
        model.config.use_cache = True
        outs: list[str] = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            outs.append(text.strip())
        return outs

    probe_prompts = [build_probe_prompt(p, default_system=default_system) for p in probes]
    max_new = probecfg["max_new_tokens"]

    # BEFORE: policy adapter still holds the unmodified SFT weights.
    before_out: list[str] | None = None
    try:
        model.set_adapter(policy_name)
        before_out = generate(probe_prompts, max_new)
        logger.info("captured %d pre-DPO (SFT) probe generations", len(before_out))
    except Exception as e:  # noqa: BLE001 -- behaviour diff is secondary to the adapter
        logger.warning("pre-DPO probe generation failed (continuing to train): %s", e)

    # 6. Experiment logging backend (same policy as SFT; loss is always recorded).
    import importlib.util as _ilu

    def _installed(pkg: str) -> bool:
        return _ilu.find_spec(pkg) is not None

    if os.environ.get("WANDB_API_KEY") and _installed("wandb"):
        report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", cfg["logging"]["wandb_project"])
    elif _installed("tensorboard") or _installed("tensorboardX"):
        report_to = ["tensorboard"]
    else:
        report_to = []
    logger.info("logging backend: %s", report_to or "disabled (loss still in trainer_state)")

    bf16 = bool(dcfg["bf16"]) and torch.cuda.is_bf16_supported()
    step_kwargs = (
        {"max_steps": cfg["smoke"]["max_steps"], "logging_steps": 1, "save_strategy": "no"}
        if args.smoke
        else {
            "num_train_epochs": dcfg["num_train_epochs"],
            "logging_steps": dcfg["logging_steps"],
            "save_strategy": dcfg["save_strategy"],
        }
    )

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=dcfg["beta"],
        loss_type=dcfg["loss_type"],
        per_device_train_batch_size=dcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dcfg["gradient_accumulation_steps"],
        learning_rate=dcfg["learning_rate"],
        lr_scheduler_type=dcfg["lr_scheduler_type"],
        warmup_ratio=dcfg["warmup_ratio"],
        optim=dcfg["optim"],
        weight_decay=dcfg["weight_decay"],
        max_grad_norm=dcfg["max_grad_norm"],
        max_length=dcfg["max_length"],
        max_prompt_length=dcfg["max_prompt_length"],
        bf16=bf16,
        fp16=not bf16,
        seed=seed,
        report_to=report_to,
        logging_dir=cfg["logging"]["tensorboard_dir"],
        gradient_checkpointing=dcfg["gradient_checkpointing"],
        # ref = frozen SFT adapter (shared base, no extra full model).
        model_adapter_name=policy_name,
        ref_adapter_name=ref_name,
        **step_kwargs,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # shared-weight reference via ref_adapter_name
        args=dpo_args,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
    )

    # 7. Train, tracking peak VRAM.
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    peak_gb = torch.cuda.max_memory_reserved() / 1e9
    logger.info("peak VRAM (reserved): %.2f GB", peak_gb)

    # 8. Save the policy adapter ONLY (standalone, M6-mergeable). Done before the
    #    after-probe pass so a generation hiccup can't lose the trained adapter.
    output_dir.mkdir(parents=True, exist_ok=True)
    model.set_adapter(policy_name)
    model.save_pretrained(str(output_dir), selected_adapters=[policy_name])
    _normalize_adapter_dir(output_dir, policy_name)
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()  # writes output_dir/trainer_state.json (loss/reward history)

    # 8b. Final reward diagnostics (used by both the conclusion and the manifest).
    state_path = output_dir / "trainer_state.json"
    final = extract_reward_metrics(json.loads(state_path.read_text(encoding="utf-8")))
    final_margin = final["rewards/margins"][-1][1] if final["rewards/margins"] else None
    final_acc = final["rewards/accuracies"][-1][1] if final["rewards/accuracies"] else None

    # 9. AFTER: same policy adapter now holds the DPO-updated weights.
    after_out: list[str] | None = None
    if before_out is not None:
        try:
            after_out = generate(probe_prompts, max_new)
            n_changed = count_changed(before_out, after_out)
            conclusion = (
                f"Training objective converged: reward margin "
                f"{f'{final_margin:.3f}' if final_margin is not None else 'n/a'}, "
                f"accuracy {f'{final_acc:.2f}' if final_acc is not None else 'n/a'} "
                f"(see dpo_margins.png). Under greedy decoding, {n_changed}/{len(probes)} "
                "probe continuations changed after DPO -- inspect the per-probe diffs "
                "above for whether pushy / rate-hallucination behaviour visibly receded."
            )
            diff_md = render_behavior_diff(
                probes, before_out, after_out,
                meta={
                    "base": mcfg["name"], "ref": "SFT (two-adapter)",
                    "beta": dcfg["beta"], "lr": f"{dcfg['learning_rate']:.0e}",
                    "pairs": len(pairs), "smoke": args.smoke,
                },
                conclusion=conclusion,
            )
            diff_path = Path(cfg["report"]["behavior_diff_path"])
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text(diff_md, encoding="utf-8")
            logger.info("wrote behaviour diff -> %s (%d probes)", diff_path, len(probes))
        except Exception as e:  # noqa: BLE001
            logger.warning("post-DPO probe generation/diff failed: %s", e)

    # 10. Loss + reward-margin curves from trainer_state.json.
    try:
        plot_loss_curve(
            state_path, Path(cfg["report"]["loss_curve_path"]), title="DPO training loss"
        )
        plot_reward_margins(state_path, Path(cfg["report"]["margins_curve_path"]))
        logger.info("wrote loss + margins curves")
    except (ValueError, FileNotFoundError) as e:
        logger.warning("could not render DPO curves: %s", e)

    # 11. Manifest + one-line summary.
    stats = {
        "n_pairs": len(pairs),
        "n_probes": len(probes),
        "smoke": args.smoke,
        "global_steps": int(trainer.state.global_step),
        "peak_vram_gb": round(peak_gb, 3),
        "final_train_loss": train_result.training_loss,
        "final_reward_margin": final_margin,
        "final_reward_accuracy": final_acc,
        "beta": dcfg["beta"],
        "ref_scheme": refcfg["scheme"],
        "base_load_in_4bit": mcfg["load_in_4bit"],
        "bf16": bf16,
        "train_runtime_s": train_result.metrics.get("train_runtime"),
    }
    inputs = [pref_path]
    sft_weights = sft_adapter / "adapter_model.safetensors"
    if sft_weights.exists():
        inputs.append(sft_weights)
    manifest = build_manifest(inputs=inputs, config=cfg, stats=stats)
    write_manifest(output_dir, manifest)

    logger.info(
        "DPO done -> adapter=%s | steps=%d | train_loss=%.4f | final_margin=%s | peak=%.2fGB",
        output_dir, trainer.state.global_step, train_result.training_loss,
        f"{final_margin:.4f}" if final_margin is not None else "n/a", peak_gb,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
