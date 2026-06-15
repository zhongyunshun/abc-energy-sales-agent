"""M4 thin CLI: Unsloth QLoRA SFT of Qwen3-4B-Instruct with completion-only loss.

Usage (GPU train container ONLY -- design doc section 5.1/5.2):
    docker compose -f docker/compose.yaml run --rm train \
        uv run python scripts/training/train_sft.py --config configs/sft.yaml [--smoke]

Pipeline (design doc section 3-M4):
  1. load + validate train/val DialogueRecords (M3 products);
  2. render each via the tokenizer's Qwen chat template (real template; the
     marker-based masking logic is unit-tested against formatting.render_chatml);
  3. Unsloth 4-bit load + LoRA; SFTTrainer; completion-only masking via
     Unsloth train_on_responses_only (locked path -- TRL 0.23.0 / Unsloth
     2025.11.1; see formatting.py / report);
  4. per-step eval val loss; W&B logging (auto-downgrade to TensorBoard when
     WANDB_API_KEY is absent -- loss is always recorded);
  5. save adapter + trainer_state.json + manifest (with peak VRAM) + loss PNG.

Heavy GPU deps (torch/unsloth/trl) are imported lazily inside main() so this file
imports cleanly on a CPU-only host. Exit codes: 0 success, 2 input-contract
failure, 3 external dependency failure (GPU/import/CUDA unavailable).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.common.schema import DialogueRecord, validate_dialogue
from sales_agent.training.formatting import to_conversation
from sales_agent.training.plotting import plot_loss_curve

logger = logging.getLogger("train_sft")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_DEPENDENCY = 3


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
    parser.add_argument("--smoke", action="store_true", help="tiny run (max_steps, n_samples)")
    parser.add_argument("--output-dir", default=None, help="override config train.output_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg, lcfg, tcfg = cfg["model"], cfg["lora"], cfg["train"]
    dcfg, maskcfg = cfg["data"], cfg["masking"]
    output_dir = Path(args.output_dir or tcfg["output_dir"])
    seed = cfg["seed"]

    # 1. Load + validate inputs (exit 2 on contract failure).
    train_path, val_path = Path(dcfg["train_path"]), Path(dcfg["val_path"])
    train_recs, val_recs = [], []
    for path, bucket in ((train_path, train_recs), (val_path, val_recs)):
        if not path.exists():
            logger.error("missing input %s (run M3 split or provide fixtures)", path)
            return EXIT_CONTRACT
        recs, errors = load_records(path)
        if errors:
            for e in errors[:10]:
                logger.error("invalid record: %s", e)
            logger.error("%s: %d invalid records (contract failure)", path.name, len(errors))
            return EXIT_CONTRACT
        bucket.extend(recs)
    if args.smoke:
        n = cfg["smoke"]["n_samples"]
        train_recs = train_recs[:n]
        val_recs = val_recs[: max(8, n // 8)]
    logger.info("loaded train=%d val=%d (smoke=%s)", len(train_recs), len(val_recs), args.smoke)

    # 2. Lazy GPU imports (exit 3 if the stack/GPU is unavailable).
    try:
        # Unsloth MUST be imported before trl/transformers so its optimizations
        # patch correctly; keep this order (sorting the block would break it).
        import torch  # noqa: I001
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        logger.error("GPU training stack unavailable (run inside the train container): %s", e)
        return EXIT_DEPENDENCY
    if not torch.cuda.is_available():
        logger.error("CUDA not available -- M4 must run in the GPU train container")
        return EXIT_DEPENDENCY

    # 3. Load base (4-bit) + attach LoRA.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg["max_seq_length"],
        load_in_4bit=mcfg["load_in_4bit"],
        dtype=mcfg.get("dtype"),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"],
        target_modules=lcfg["target_modules"],
        use_gradient_checkpointing=lcfg["use_gradient_checkpointing"],
        use_rslora=lcfg["use_rslora"],
        random_state=lcfg["random_state"],
    )

    # 4. Render with the REAL Qwen chat template (not formatting.render_chatml,
    #    which mirrors it only for unit tests). Build text datasets.
    default_system = dcfg.get("default_system")

    def to_text(rec: DialogueRecord) -> dict:
        conv = to_conversation(rec, default_system=default_system)
        text = tokenizer.apply_chat_template(
            conv["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_ds = Dataset.from_list([to_text(r) for r in train_recs])
    eval_ds = Dataset.from_list([to_text(r) for r in val_recs]) if val_recs else None

    # 5. Experiment logging backend selection. Preference: W&B (keyed + installed)
    #    -> local TensorBoard (installed) -> none. This only controls the optional
    #    live dashboard: the loss history is ALWAYS captured in trainer_state.json
    #    (-> loss PNG + manifest) independent of report_to, so "never skip loss"
    #    holds even when the locked GPU image ships no logging-backend package
    #    (it currently ships neither wandb nor tensorboard).
    import importlib.util as _ilu

    def _installed(pkg: str) -> bool:
        return _ilu.find_spec(pkg) is not None

    if os.environ.get("WANDB_API_KEY") and _installed("wandb"):
        report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", cfg["logging"]["wandb_project"])
        logger.info("logging to W&B project=%s", cfg["logging"]["wandb_project"])
    elif _installed("tensorboard") or _installed("tensorboardX"):
        report_to = ["tensorboard"]
        logger.info("no W&B -> TensorBoard at %s", cfg["logging"]["tensorboard_dir"])
    else:
        report_to = []
        logger.info(
            "no wandb/tensorboard package -> external reporting disabled; loss is "
            "still recorded in trainer_state.json and %s",
            cfg["report"]["loss_curve_path"],
        )

    bf16 = bool(tcfg["bf16"]) and torch.cuda.is_bf16_supported()

    # Optional torch.compile (opt-in via config). On a big card the spare memory
    # can be traded for throughput; Unsloth already fuses hot kernels, so compile
    # is supplementary -- validate the actual speedup on --smoke before relying on
    # it (it can graph-break against Unsloth's patches). Only the resulting adapter
    # matters for M5/M6 and compilation does not change it numerically.
    compile_kwargs: dict = {}
    if tcfg.get("torch_compile", False):
        compile_kwargs["torch_compile"] = True
        if tcfg.get("torch_compile_backend"):
            compile_kwargs["torch_compile_backend"] = tcfg["torch_compile_backend"]
        if tcfg.get("torch_compile_mode"):
            compile_kwargs["torch_compile_mode"] = tcfg["torch_compile_mode"]
        logger.info("torch.compile enabled (backend=%s mode=%s)",
                    tcfg.get("torch_compile_backend", "default"),
                    tcfg.get("torch_compile_mode", "default"))

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        optim=tcfg["optim"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        max_length=mcfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        bf16=bf16,
        fp16=not bf16,
        seed=seed,
        report_to=report_to,
        logging_dir=cfg["logging"]["tensorboard_dir"],
        dataset_num_proc=2,
        **compile_kwargs,
        # smoke vs full: cap steps and tighten eval/log cadence for a fast check.
        **(
            {
                "max_steps": cfg["smoke"]["max_steps"],
                "logging_steps": 1,
                "eval_strategy": "steps" if eval_ds is not None else "no",
                "eval_steps": 5,
                "save_steps": cfg["smoke"]["max_steps"],
                "save_total_limit": 1,
            }
            if args.smoke
            else {
                "num_train_epochs": tcfg["num_train_epochs"],
                "logging_steps": tcfg["logging_steps"],
                "eval_strategy": tcfg["eval_strategy"] if eval_ds is not None else "no",
                "eval_steps": tcfg["eval_steps"],
                "save_strategy": tcfg["save_strategy"],
                "save_steps": tcfg["save_steps"],
                "save_total_limit": tcfg["save_total_limit"],
            }
        ),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    # 6. Completion-only masking: loss on assistant turns only (locked path).
    trainer = train_on_responses_only(
        trainer,
        instruction_part=maskcfg["instruction_part"],
        response_part=maskcfg["response_part"],
    )

    # 7. Train, tracking peak VRAM.
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    peak_gb = torch.cuda.max_memory_reserved() / 1e9
    logger.info("peak VRAM (reserved): %.2f GB", peak_gb)
    if peak_gb > 10.0:
        logger.warning(
            "peak VRAM %.2f GB exceeds the 10GB budget -- report before scaling up", peak_gb
        )

    # 8. Save adapter + trainer_state.json; render loss curve.
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()  # writes output_dir/trainer_state.json (loss history)

    state_path = output_dir / "trainer_state.json"
    loss_png = Path(cfg["report"]["loss_curve_path"])
    try:
        plot_loss_curve(state_path, loss_png, title="SFT training loss")
        logger.info("wrote loss curve -> %s", loss_png)
    except (ValueError, FileNotFoundError) as e:
        logger.warning("could not render loss curve: %s", e)

    eval_metrics = trainer.evaluate() if eval_ds is not None else {}
    stats = {
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "smoke": args.smoke,
        "global_steps": int(trainer.state.global_step),
        "peak_vram_gb": round(peak_gb, 3),
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_metrics.get("eval_loss"),
        "bf16": bf16,
        "torch_compile": bool(compile_kwargs.get("torch_compile", False)),
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "masking": "train_on_responses_only",
    }
    manifest = build_manifest(inputs=[train_path, val_path], config=cfg, stats=stats)
    write_manifest(output_dir, manifest)

    logger.info(
        "SFT done -> adapter=%s | steps=%d | train_loss=%.4f | eval_loss=%s | peak=%.2fGB",
        output_dir, trainer.state.global_step, train_result.training_loss,
        f"{eval_metrics.get('eval_loss'):.4f}" if eval_metrics.get("eval_loss") else "n/a",
        peak_gb,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
