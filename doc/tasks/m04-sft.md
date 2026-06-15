# M4 SFT Training — Task List

> English snapshot of `doc/tasks/m04-sft.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Design refs: `detailed-design.md` §3-M4 | Prerequisites: M0, M3 products (or fixtures), container GPU available (T0.8)

## Tasks

- [x] T4.1 Implement `training/formatting.py`: `DialogueRecord` → Qwen chat template rendering + completion-only masking boundaries (confirm the API for the locked TRL version, see design §6-1) + unit tests (assistant-span start/end positions, multi-turn masking correctness)
- [x] T4.2 Author `configs/sft.yaml`: put the full hyperparameter baseline table from design §3-M4 into config (LoRA r16/α32, lr 2e-4 cosine, bs 2×8, seq 2048, adamw_8bit, gradient_checkpointing)
- [x] T4.3 Author `scripts/training/train_sft.py`: Unsloth 4-bit load → SFTTrainer → step-wise val-loss eval → W&B reporting (auto-degrade to TensorBoard without a key) → save adapter + manifest
- [x] T4.4 Implement loss-curve export: plot a PNG from `trainer_state.json` to `reports/training/sft_loss.png` (the plotting function is unit-testable)
- [x] T4.5 Container `--smoke` run (max_steps=10, 64 samples): peak VRAM ≤10GB, adapter reloadable by PEFT for generation
- [x] T4.6 Full training (2 epochs): loss converges, val loss does not diverge; manually compare base vs SFT outputs on 5 prompts to confirm the style change

## Definition of Done (DoD)

formatting unit tests green; both smoke and full produce a valid adapter; loss-curve PNG produced (report material).
