# M7 Quantization (AWQ) — Task List

> English snapshot of `doc/tasks/m07-quant.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Design refs: `detailed-design.md` §3-M7 | Prerequisites: M0, M6 merged model

## Tasks

- [x] T7.1 Implement calibration-set construction: stratified-sample 256 from train.jsonl → render plain text via the chat template (reuse formatting.py) + unit tests (count, stratification, render format)
- [x] T7.2 Author `configs/quant.yaml` + `scripts/quant/quantize_awq.py`: llm-compressor AWQ recipe (W4A16, group_size 128) → output `models/quantized/awq/`; `--smoke` uses 32 calibration samples
- [x] T7.3 Product load self-check: transformers loads the quantized model + a single generation succeeds; size stats (FP16 vs INT4 GB) written to the manifest
- [x] T7.4 Run full quantization on the real merged model in the container; manually compare 5 prompts vs FP16 output to confirm no visible regression (the formal comparison is left to M9/M10)

## Definition of Done (DoD)

The AWQ product is recognized and loaded by vLLM (verified in M8); size-comparison data feeds the README trade-off material.
