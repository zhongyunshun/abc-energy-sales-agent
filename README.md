# Sales-Conversation LLM — Full-Pipeline R&D

End-to-end pipeline that turns `Qwen3-4B-Instruct-2507` into a deployable INT4
telesales / energy-sales assistant: data engineering → SFT → DPO → adapter merge
→ AWQ quantization → vLLM serving → offline + LLM-judge evaluation → load test.

- **Base model:** `unsloth/Qwen3-4B-Instruct-2507`
- **Final artifact:** INT4 AWQ (compressed-tensors W4A16) model served by vLLM on
  a single RTX 4070 12GB, OpenAI-compatible endpoint.
- **Technical / performance report:** [`reports/REPORT.md`](reports/REPORT.md)
  answers the challenge section by section (A data, B fine-tuning, C quant/serving,
  D evaluation, Bonus) with all loss curves, benchmark plots, and evaluation tables
  (each annotated with its source file). This README covers tech choices, how to
  run, and honest disclosures.
- **Challenge coverage:** §10 maps every requirement in the challenge brief
  (`mini_test_2.pdf`) to the code and docs that answer it.

---

## 1. Pipeline at a glance

| Stage                                    | Module | Script | Where it ran |
|------------------------------------------|---|---|---|
| Normalize public data                    | M1 | [`scripts/data/normalize.py`](scripts/data/normalize.py) | CPU |
| Synthesize dialogues + DPO pairs         | M2 | [`scripts/data/synthesize.py`](scripts/data/synthesize.py) | API (OpenRouter) |
| Split + leakage check                    | M3 | [`scripts/data/split.py`](scripts/data/split.py) | CPU |
| SFT (LoRA)                               | M4 | [`scripts/training/train_sft.py`](scripts/training/train_sft.py) | **A100 40GB (bf16)** |
| DPO alignment                            | M5 | [`scripts/training/train_dpo.py`](scripts/training/train_dpo.py) | **A100 40GB (bf16)** |
| Merge adapter → dense                    | M6 | [`scripts/training/merge_adapter.py`](scripts/training/merge_adapter.py) | A100 / CPU |
| AWQ quantization                         | M7 | [`scripts/quant/quantize_awq.py`](scripts/quant/quantize_awq.py) | **RTX 4070 (CPU-offload)** |
| vLLM serving                             | M8 | [`scripts/serving/serve.sh`](scripts/serving/serve.sh) | **RTX 4070** |
| Offline rule eval                        | M9 | [`scripts/eval/run_offline_eval.py`](scripts/eval/run_offline_eval.py) | **RTX 4070** (endpoint) |
| LLM-as-a-Judge                           | M10 | [`scripts/eval/run_judge.py`](scripts/eval/run_judge.py) | API (OpenRouter) |
| Load test (Locust)                       | M11 | [`scripts/bench/run_bench.py`](scripts/bench/run_bench.py) | **RTX 4070** (endpoint) |
| 1,000+ concurrency blueprint + KV Cache  | Bonus 1 | [`Bonus/bonus1_capacity/kv_capacity_calc.py`](Bonus/bonus1_capacity/kv_capacity_calc.py) | CPU (paper sizing) |
| Advanced PTQ + smoke test                | Bonus 2 | [`Bonus/bonus2_quant/quantize_modelopt.py`](Bonus/bonus2_quant/quantize_modelopt.py) | **RTX 4070** (Docker / ModelOpt) |
| Auto benchmarking + observability traces | Bonus 3 | [`Bonus/bonus3_observability/run_multiturn_trace_demo.py`](Bonus/bonus3_observability/run_multiturn_trace_demo.py) | **RTX 4070** endpoint + local/Langfuse |

The table above is the module DAG; the sections below cover requirements, execution details, disclosures, and acceptance evidence.

Bonus 1-3 run steps are documented in their folder READMEs:
[`Bonus 1`](Bonus/bonus1_capacity/README.md),
[`Bonus 2`](Bonus/bonus2_quant/README.md), and
[`Bonus 3`](Bonus/bonus3_observability/README.md).

---

## 2. Tech choices & rationale

### Model & LoRA
- **`Qwen3-4B-Instruct-2507`** — a 4B instruct model fits the 12GB serving target
  after INT4 quantization with room for KV cache, while being strong enough to
  learn sales persona and objection handling.
- **LoRA r32 / α64** (all 7 attention+MLP projection modules), chosen by an
  explicit hyperparameter sweep — r32/α64 @ lr2e-4 reached val loss **0.9496** vs
  **0.9720** for the r16/α32 baseline (`reports/training/sft_hparam_sweep.md`).
  This is *not* the original r16/α32 default.

### Alignment
- **SFT then DPO.** SFT teaches the persona and short voice-style replies; DPO
  targets two failure modes (pushy closes, invented rates) via 300 preference
  pairs. DPO uses a **two-adapter reference** (frozen `reference` adapter = SFT, so
  KL is anchored to the SFT policy, not the raw base).

### Quantization — AWQ W4A16 (compressed-tensors)
- **Main route: AWQ INT4 / W4A16_ASYM** via `llm-compressor`, calibrated on
  scenario-stratified training data. Chosen for the best size/quality trade-off on
  a 12GB card (3.018× smaller, no visible quality regression — see §4).
- The output is **compressed-tensors / pack-quantized** format, which vLLM
  auto-detects (no `--quantization` flag needed). GGUF Q4_K_M is noted as the
  edge/CPU alternative but not run here.

### Serving — vLLM
- Official `vllm/vllm-openai:v0.10.2` image (no self-built serving image),
  OpenAI-compatible API, continuous batching, prefix caching. 12GB-specific tuning
  is in §5.2 below.

---

## 3. How to run (uv and Docker)

Two execution surfaces are used:

- **uv (host)** — CPU stages (M1, M3), API stages (M2, M10), serving control
  (M8 [`serve.sh`](scripts/serving/serve.sh)), and endpoint clients (M9, M11). The host installs only the
  `dev + api + bench` dependency groups. **No GPU Python packages on the host.**
- **Docker (train image)** — the GPU stack (M4–M7). `torch`, `unsloth`, `trl`,
  `llmcompressor` live **only** inside `docker/Dockerfile.train` (`--group gpu`).
- **vLLM (official image)** — serving (M8) uses `vllm/vllm-openai:v0.10.2`
  directly; nothing is self-built.

Convention used below: a **Docker** form for any GPU stage is
`docker compose -f docker/compose.yaml run --rm train -lc "<cmd>"`. The train
image's entrypoint is `bash` with the project venv on `PATH`, so `<cmd>` calls
`python ...` directly. Every stage script accepts `--smoke` for a fast pipeline
connectivity check before the full run; `--config` is required; `--output-dir`
overrides the config's output location.

### 3.0 Prerequisites & environment

```bash
# 1. uv (the package manager). Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh        # or: pipx install uv

# 2. Host deps (CPU/API/bench groups; uv provisions the pinned Python from
#    .python-version automatically). Run from the repo root:
uv sync                                                 # -> .venv with dev+api+bench

# 3. Secrets: create a gitignored .env at the repo root with the keys read at
#    runtime (never commit it):
#      OPENROUTER_API_KEY=sk-or-...   # REQUIRED for M2 (synthesis) and M10 (judge)
#      WANDB_API_KEY=...              # OPTIONAL; M4/M5 fall back to TensorBoard/none

# 4. Sanity-check the host install (GPU tests are deselected by default):
uv run pytest -q
```

For the GPU stack you also need an NVIDIA driver + the NVIDIA Container Toolkit
(so `docker run --gpus all` works), then build the training image once:

```bash
docker compose -f docker/compose.yaml build train       # CUDA 12.4 runtime + uv + group `gpu`
```

> **A100 (no-Docker) path.** M4/M5 were actually run on a shared A100 node without
> docker-group permission, using an isolated uv venv instead of the container:
> `uv venv && uv sync --group gpu` then the same `python scripts/...` commands.
> This is a documented deviation from the "Docker for GPU" standard, recorded on 2026-06-13; the container path is the portable default.

> **GPU sharing rule.** Training and serving share a single card and
> must **never run at the same time** — stages are serial. [`serve.sh`](scripts/serving/serve.sh) warns if a
> `sales-agent-train` container is up. Enforcement is operational, not automatic
> (see §6).

### 3.1 Quick start (committed artifacts → served model → eval)

`data/` and `reports/` are committed, so a fresh checkout can skip M1–M7 and go
straight to serving the (gitignored) model — or you can regenerate any stage. To
go end-to-end **from a checkout that already has `models/quantized/awq/`**:

```bash
bash scripts/serving/serve.sh                                    # M8: up + health
uv run python scripts/serving/concurrency_demo.py               # M8: batching demo
uv run python scripts/eval/run_offline_eval.py \
    --config configs/eval_offline.yaml --model-tag dpo          # M9 (deployed model)
uv run python scripts/eval/run_judge.py --config configs/eval_judge.yaml   # M10
uv run python scripts/bench/run_bench.py --config configs/bench.yaml       # M11
```

The full ordered pipeline is M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → {M9, M11} →
M10. Per-stage detail follows.

### 3.2 Data engineering (M1–M3) — host / API

**M1 — Normalize** (CPU). Downloads `goendalf666/sales-conversations` at runtime,
explodes `prefixed_pairs` rows into single-turn `DialogueRecord`s, cleans
(PII scrub, langdetect, dedup), tags scenarios.
```bash
uv run python scripts/data/normalize.py --config configs/normalize.yaml          # full
uv run python scripts/data/normalize.py --config configs/normalize.yaml --smoke  # 50/source
```
- reads: HF dataset (cached) · writes: `data/interim/normalized.jsonl`,
  `normalize_report.json`, `manifest.json`
- verify: `normalize_report.json` per-source written/dropped counts

**M2 — Synthesize** (OpenRouter; needs `OPENROUTER_API_KEY`). Two modes; run both.
Model is config-driven (`model:` in `synthesize.yaml`, default
`google/gemini-2.5-flash`) and overridable with `--model`.
```bash
uv run python scripts/data/synthesize.py --config configs/synthesize.yaml --mode dialogues
uv run python scripts/data/synthesize.py --config configs/synthesize.yaml --mode preferences
# --smoke generates 2/scenario for a cheap connectivity check first.
```
- writes: `data/interim/synthetic_dialogues.jsonl`, `preference_pairs.jsonl`,
  and `*_cost_report.json` (raw token counts + USD estimate)
- verify: cost report `succeeded`/`abandoned`; committed run = 4,351 dialogues + 300 pairs

**M3 — Split + leakage check** (CPU). Merges M1+M2 → exact dedup → MinHash
near-dedup → M1 downsample (2:1) → stratified split → leakage assertion.
```bash
uv run python scripts/data/split.py --config configs/split.yaml          # full
uv run python scripts/data/split.py --config configs/split.yaml --smoke  # capped
```
- reads: `normalized.jsonl`, `synthetic_dialogues.jsonl`
- writes: `data/processed/{train,val,test}.jsonl`, `split_report.json`
- verify (acceptance §8-3): `split_report.json` →
  `leakage_check.cross_split_dups == 0` (MinHash @ 0.85)

Docker form (the train image also has the `api` group), e.g.:
`docker compose -f docker/compose.yaml run --rm train -lc "python scripts/data/split.py --config configs/split.yaml"`.

### 3.3 Training & alignment (M4–M6) — GPU

Run via the train container, e.g.
`docker compose -f docker/compose.yaml run --rm train -lc "<cmd>"`, or the A100
uv-venv path (§3.0). **Always `--smoke` first.**

**M4 — SFT (LoRA)** — A100 40GB bf16.
```bash
# Full (A100, bf16, r32/α64, batch 16):
python scripts/training/train_sft.py --config configs/sft_a100.yaml
# 4070 connectivity smoke (4-bit QLoRA, batch 2):
python scripts/training/train_sft.py --config configs/sft.yaml --smoke
# Optional LoRA sweep (picks r32/α64):
python scripts/training/sweep_sft.py --base-config configs/sft_a100.yaml
```
- `sft_a100.yaml` = `load_in_4bit=false`, batch 16; `sft.yaml` = 4070 4-bit smoke
  profile (`load_in_4bit=true`, batch 2)
- reads: `data/processed/{train,val}.jsonl` · writes: `models/adapters/sft/`,
  `reports/training/sft_loss.png`, `sft_manifest.json`, `sft_trainer_state.json`
- verify / expected: `sft_manifest.json` `final_eval_loss ≈ 0.84`, peak ~17 GB,
  ~42 min on A100

**M5 — DPO** — A100 (needs ≥24 GB; peaked ~27.9 GB). Continues the SFT adapter on
300 preference pairs with a two-adapter reference.
```bash
python scripts/training/train_dpo.py --config configs/dpo.yaml --smoke   # 10 steps first
python scripts/training/train_dpo.py --config configs/dpo.yaml           # full
```
- reads: `data/interim/preference_pairs.jsonl`, `models/adapters/sft/`
- writes: `models/adapters/dpo/`, `reports/training/dpo_loss.png`,
  `dpo_margins.png`, `dpo_behavior_diff.md`, `dpo_manifest.json`
- verify / expected: reward margin → 1.157, accuracy 1.00; ~64 s

**M6 — Merge adapter → dense BF16.** Folds the DPO LoRA into the base; runs an
8-prompt greedy consistency check.
```bash
python scripts/training/merge_adapter.py --config configs/merge.yaml --adapter models/adapters/dpo
```
- writes: merged model dir + `reports/training/merge_consistency.md`,
  `merge_manifest.json`
- verify: `merge_consistency.md` → **PASS, 8/8**
- ⚠️ path note: `merge.yaml` `output_dir` is `models/merged` (contract path), but on
  the PC the merged model was placed at `models/adapters/merged`; M7 points there
  via `quant.yaml model.merged_dir` (override with `--merged-dir`). See §7.

### 3.4 Quantization (M7) — RTX 4070

AWQ INT4 / W4A16 via `llm-compressor`, run on the 4070 with `device_map=cpu` +
sequential per-block calibration (one decoder block on the GPU at a time, so peak
GPU memory is a few GB, never the whole ~8 GB model).
```bash
python scripts/quant/quantize_awq.py --config configs/quant.yaml --smoke   # 32-sample calib
python scripts/quant/quantize_awq.py --config configs/quant.yaml           # 256-sample calib
```
- reads: merged model (`quant.yaml model.merged_dir`), `data/processed/train.jsonl`
  (calibration) · writes: `models/quantized/awq/` (compressed-tensors),
  `reports/training/quant_manifest.json`, `quant_report.md`
- verify / expected: `quant_manifest.json` `sizes` → 8.045 → 2.666 GB (3.018×),
  `self_check_ok: true`; report has 5 FP16-vs-INT4 probes

### 3.5 Serving (M8) — RTX 4070, vLLM

```bash
bash scripts/serving/serve.sh                          # recommended: pre-flight + patch + health poll
# standalone equivalent (no pre-flight / health poll):
docker compose -f docker/compose.yaml up serve
```
[`serve.sh`](scripts/serving/serve.sh) (reads everything from `configs/serve.yaml`): (1) asserts
`models/quantized/awq/` exists — exit 2 if not (run M7); (2) checks Docker is up —
exit 3 if not; (3) warns if a train container shares the GPU; (4) runs the
idempotent [`patch_quant_config.py`](scripts/serving/patch_quant_config.py) (vLLM v0.10.2 compat, §5.2); (5)
`docker compose up -d serve`; (6) polls `/health` until 200 or the 120 s timeout
(exit 3 on timeout, dumping recent logs). On success: endpoint
`http://127.0.0.1:8000/v1`, served model `sales-agent-awq`.

Continuous-batching demo (16 concurrent streams):
```bash
uv run python scripts/serving/concurrency_demo.py --config configs/serve.yaml --n 16
```
writes `reports/serving/concurrency_demo.{json,md}` (expected ~10× latency speedup).

### 3.6 Evaluation & load test (M9–M11)

The endpoint from §3.5 must be healthy. M9/M11 hit the 4070 endpoint; M10 calls
OpenRouter.

**M9 — Offline rule eval.** Run once **per model group**, same config so all three
see the identical 650-row test batch.
```bash
uv run python scripts/eval/run_offline_eval.py --config configs/eval_offline.yaml --model-tag dpo
uv run python scripts/eval/run_offline_eval.py --config configs/eval_offline.yaml --model-tag base
uv run python scripts/eval/run_offline_eval.py --config configs/eval_offline.yaml --model-tag sft
```
- ⚠️ base/sft are different weights than the deployed DPO model: to evaluate them
  you re-serve each group's AWQ build (or use the local-inference fallback flags
  `--local-model` / `--local-adapter`). `--smoke` runs 10 rows first.
- writes: `reports/eval_offline/<tag>/{results.jsonl,summary.json,manifest.json}`
  and the cross-group `comparison.md`

**M10 — LLM-as-a-Judge** (OpenRouter; two non-Google judges from
`eval_judge.yaml`). Scores the same 100 ids/group from M9's outputs, blind.
```bash
uv run python scripts/eval/run_judge.py --config configs/eval_judge.yaml --smoke   # 5/group
uv run python scripts/eval/run_judge.py --config configs/eval_judge.yaml           # full 100/group
```
- reads: the three `reports/eval_offline/<tag>/results.jsonl`
- writes: `reports/eval_judge/{scores.jsonl,aggregate.json,comparison.md,manifest.json}`
- expected cost: ~$4.20 (recorded in `manifest.json`)

**M11 — Load test** (Locust ladder `[1,4,8,16,32]`, ~12 min full).
```bash
uv run python scripts/bench/run_bench.py --config configs/bench.yaml --smoke   # 30 s, 1 tier
uv run python scripts/bench/run_bench.py --config configs/bench.yaml           # full ladder
uv run python scripts/bench/plot_bench.py --config configs/bench.yaml          # re-plot only
```
- writes: `reports/bench/{raw_<c>.csv, bench_summary.csv, *_vs_concurrency.png (×3),
  bench_report.md, manifest.json}`
- expected: 0 % error across tiers, knee at 16→32 (see §4 / `reports/REPORT.md` §6)

---

## 4. Quantization trade-off

Source: `reports/training/quant_manifest.json` (M7) + `reports/bench/bench_report.md` (M11).

| | FP16 (merged) | INT4 (AWQ) |
|---|---:|---:|
| size | 8.045 GB | **2.666 GB** |
| ratio | — | **3.018× smaller (66.86 % reduction)** |
| fits on 4070 12GB + display? | no (~7.9 GB free) | **yes** |
| quality (5 greedy probes) | reference | **no visible regression** |

- **Why INT4 is the deployed artifact:** the FP16 merged model (8.045 GB) does not
  fit alongside the display on the 4070 (~7.9 GB free); INT4 (2.666 GB) leaves
  ample room for KV cache and continuous batching. vLLM loads the
  compressed-tensors weights at ~2.57 GB.
- **Throughput vs concurrency** (INT4 endpoint, `reports/bench/`): scales
  near-linearly to 102→820 tok/s from 1→16 concurrency, then the knee at 16→32
  (32 > `max_num_seqs=16`, vLLM queues) drives TTFT p50 112→868 ms while throughput
  plateaus. 0 % error across all tiers.
- **FP16-vs-INT4 latency was not A/B-measured** (FP16 won't co-reside with the
  display on the 4070); the comparison rests on size (3.018×) + the 5 quality
  probes. This gap is disclosed in §7 and `reports/bench/bench_report.md`.

Full plots and tables: [`reports/REPORT.md`](reports/REPORT.md) §4, §6.

---

## 5. Serving configuration (12GB tuning)

`configs/serve.yaml` is the single source of truth; `docker/compose.yaml` mirrors
the engine flags (kept in sync by `tests/test_serve_config.py`).

### 5.1 Measured tuning values (RTX 4070 12GB)
| flag | value | why |
|---|---|---|
| `gpu_memory_utilization` | **0.55** | the card also drives the display (~4GB used, ~7.9GB free); the original default 0.90 OOMs |
| `max_model_len` | **3072** | voice dialogues are short; small KV cache leaves headroom |
| `max_num_seqs` | **16** | continuous-batching concurrency cap (matches the 16-way demo) |
| `enable_prefix_caching` | **on** | reuse the shared system-prompt KV across requests |
| `reasoning_parser` | **qwen3** | strips the empty `<think></think>` prefix into `reasoning_content` (see §7) |
| `--quantization` | **omitted** | compressed-tensors is auto-detected; forcing `awq` (AutoAWQ) misparses and fails to load |

Steady-state VRAM ≈ 8.5 GB (vLLM ~6.6 + display ~1.8), ~3.8 GB headroom; cold start
~68 s (< the 120 s health timeout).

### 5.2 [`patch_quant_config.py`](scripts/serving/patch_quant_config.py) (vLLM v0.10.2 workaround)
The M7 product (compressed-tensors 0.16.0) writes `scale_dtype` / `zp_dtype` into
`config.json`, which vLLM v0.10.2 (pydantic `extra='forbid'`) rejects on startup.
[`scripts/serving/patch_quant_config.py`](scripts/serving/patch_quant_config.py) strips those two metadata fields before
serving. It is **idempotent** and only removes dtype *metadata* — the W4A16 kernel
reads real dtypes from the safetensors. [`serve.sh`](scripts/serving/serve.sh) runs it automatically; if M7 is
re-run the fields return and the patch re-applies cleanly. (It edits
`models/quantized/awq/config.json`, which is gitignored.)

---

## 6. Resource isolation (shared GPU)

For shared-GPU operation:

- **This project: time-slicing.** Training (M4/M5, A100) and serving (M8, 4070)
  never run concurrently; on the 4070 specifically, the quant/serve/eval/bench
  stages are run **serially** so benchmark numbers are not distorted by a competing
  workload. The compose file documents this but does not hard-enforce it.
- **Production options** (the 1,000-session scaling blueprint is worked out in
  [`Bonus/bonus1_capacity/`](Bonus/bonus1_capacity/)): NVIDIA **MIG** (hard
  partitioning on A100/H100), **MPS** (soft sharing), per-process memory
  fractions / quotas, and **Kubernetes device plugins** for multi-replica serving.

---

## 7. Honest disclosures

Every item below is backed by the committed artifacts under `reports/` and `data/`.

### Training
- **SFT (M4) and DPO (M5) ran on an A100 40GB in bf16**, *not* the 4070 4-bit
  QLoRA path. SFT: `load_in_4bit=false`, LoRA r32/α64, 2 epochs / 1,470 steps,
  peak VRAM **17.0 GB**, ~42 min (`reports/training/sft_manifest.json`). DPO: 300
  pairs / 19 steps, peak **27.9 GB** (`reports/training/dpo_manifest.json`).
- **The 4070 path was used for smoke validation only** (`configs/sft.yaml`,
  `--smoke`), confirming pipeline connectivity at ~7.23 GB. *Caveat:* the 4070
  smoke VRAM artifact was overwritten in a later git merge, so the citable
  retained training-VRAM figure is the A100 full run (17.0 GB) in
  `sft_manifest.json`.
- **DPO behavioural gain is mild and honestly reported:** the reward margin
  converged (0→1.157, accuracy 1.00), but only **5/20** greedy probes changed and
  one rate hallucination ("25 pence a day standing charge") survived. SFT was
  already well-aligned (`reports/training/dpo_behavior_diff.md`).

### Quantization (M7)
- Run on the **4070 with CPU-offload, sequential per-block** quantization;
  **W4A16_ASYM / compressed-tensors**; FP16 **8.045 GB → INT4 2.666 GB**
  (**3.018× / 66.86 % reduction**).

### Serving (M8)
- 12GB-tuned: **gpu_util 0.55** (display takes ~4GB), **max_model_len 3072**,
  **max_num_seqs 16**, prefix caching, **`--reasoning-parser qwen3`**, and
  **`--quantization awq` removed** (compressed-tensors auto-detected).
  [`patch_quant_config.py`](scripts/serving/patch_quant_config.py) is an idempotent workaround for vLLM v0.10.2 (§5.2).

### The empty `<think></think>` prefix
- The Qwen3 chat template injects `<think>\n\n</think>` into the last assistant
  turn; our data has no reasoning, so SFT learned to echo an *empty* think prefix
  (inconsistently — 3/5 base-vs-SFT probes). This is **not** a masking bug.
  **Decision = Option A (downstream strip, no retrain):** M8 uses the qwen3
  reasoning parser; M9/M10 strip the prefix before scoring so it cannot pollute
  length metrics or judge scores.

### Evaluation
- **M9:** three groups (base / SFT / SFT+DPO) all generated via the **4070 AWQ
  endpoint** (approach B), same 650-id test batch, temperature=0.
- **M10:** **two non-Google judges** (`anthropic/claude-sonnet-4.6` +
  `openai/gpt-5.4`), chosen to avoid same-source bias with the
  `google/gemini-2.5-flash` synthesizer. Cost **$4.20**. Result: SFT ≫ base on
  professionalism/coherence/sales_logic; **SFT→DPO no significant difference** —
  matching M9 and the M5 probes.
- **M11:** Locust ladder, 16→32 knee, 0 % error. **FP16-vs-INT4 latency was not
  measured** (8 GB FP16 won't fit on the 4070 alongside the display); the
  size/quality comparison points to M7 (3.018× + 5 probes). Real, recorded gap.

### Data
- **Public dataset `goendalf666/sales-conversations` declares no license** on the
  HF hub (content is GPT-3.5-synthetic, no PII). Used as base training material for
  this R&D challenge, documented here and in `data/README.md`.
- **Data retention decision:** the small (~10 MB) `data/` products **are committed** so remote runs reproduce without re-downloading.
- **M1 downsampling 2:1** (M1:M2) so 19,487 single-turn `general` records don't
  drown the M2 multi-turn energy dialogues (`data/processed/split_report.json`,
  `downsample.ratio = 2.0`).
- **M2 synthesized with `google/gemini-2.5-flash`** (a cost decision; the design
  default was `claude-sonnet`). Actual committed cost: dialogues **$10.74**
  (4,351 records, batch A $3.42 + batch B $7.32) + preference pairs **$0.22**
  (300 pairs) = **≈ $10.96** (`data/interim/*_cost_report.json`). *(An earlier
  "$3.64" estimate predates the 1,387→4,351 expansion and is superseded.)*

---

## 8. Acceptance checklist

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Clean-env reproducible full chain (data→train→merge→quant→serve→eval→bench) | ✅ | §3 commands (uv + Docker) for every stage |
| 2 | `reports/` has real loss curves, TTFT/ITL/throughput, base/SFT/DPO judge scores | ✅ | `reports/training/*.png`, `reports/bench/*`, `reports/eval_judge/comparison.md` |
| 3 | Split passes leakage check (no cross-split dup/near-dup) | ✅ | `data/processed/split_report.json`: `leakage_check.cross_split_dups = 0` (minhash@0.85) |
| 4 | Whitepaper / advanced items (1k concurrency, NVFP4, isolation, Langfuse) | ✅ | **implemented in [`Bonus/`](Bonus/)** — bonus 1 A100 capacity blueprint, bonus 2 ModelOpt FP8 PTQ, bonus 3 Langfuse tracing (see REPORT "Bonus Challenges") |

### Repository public-readiness
- `.env` is **not tracked** (gitignored); secrets read from env vars
  (`OPENROUTER_API_KEY`, `WANDB_API_KEY`).
- `models/*` gitignored (only `.gitkeep` tracked); no large model binaries in the
  repo. Largest tracked file is `data/processed/train.jsonl` (13.6 MB).
- `git grep` finds **no API-key literals** in tracked content.

---

## 9. Repository layout

```
configs/      stage configs (single source of truth) + prompt templates
data/         committed pipeline products (M1 normalized, M2 synthetic, M3 splits)
docker/       Dockerfile.train (M4-M7) + compose.yaml (train / serve)
doc/          task files
reports/      REPORT.md + training / serving / eval_offline / eval_judge / bench artifacts
scripts/      thin CLIs per module (data / training / quant / serving / eval / bench)
src/          sales_agent package: schema, io, openrouter, data, training, quant, evals, bench
tests/        unit + config-guard tests (`uv run pytest`; GPU tests marked, deselected by default)
Bonus/        bonus 1 A100 capacity blueprint · bonus 2 ModelOpt FP8 PTQ · bonus 3 Langfuse tracing
```

---

## 10. Challenge requirements → where answered

Every requirement in the ABC Energy brief (`mini_test_2.pdf`), mapped to the code
that implements it and the doc that explains it. (`REPORT.md` = the section-by-section
technical report; slides = `doc/presentation/`; bonus implementations = [`Bonus/`](Bonus/),
each with its own README + results.)

| Challenge requirement                                                     | Code | Where answered |
|---------------------------------------------------------------------------|---|---|
| **A. Normalization** (raw logs → clean multi-turn `messages`)             | [`scripts/data/normalize.py`](scripts/data/normalize.py), [`src/sales_agent/data/normalize.py`](src/sales_agent/data/normalize.py) | REPORT A.1 · README §3.2 |
| **A. Data Synthesis** (objection handling, info gathering)                | [`scripts/data/synthesize.py`](scripts/data/synthesize.py) | REPORT A.2 · README §3.2 |
| **A. Validation** (train/val/test, no leakage, representative)            | [`scripts/data/split.py`](scripts/data/split.py) | REPORT A.3 · README §8-3 |
| **B. SFT** (scalable script; LoRA rank/alpha/lr; memory efficiency)       | [`scripts/training/train_sft.py`](scripts/training/train_sft.py), [`sweep_sft.py`](scripts/training/sweep_sft.py) | REPORT B.1 · README §2, §3.3 |
| **B. Alignment / DPO** (pushy + rate hallucination)                       | [`scripts/training/train_dpo.py`](scripts/training/train_dpo.py) | REPORT B.2 |
| **B. Adapter Management** (merge LoRA → dense)                            | [`scripts/training/merge_adapter.py`](scripts/training/merge_adapter.py) | REPORT B.3 · README §3.3 |
| **C. Quantization** (PTQ + size↔latency TTFT/ITL trade-off)               | [`scripts/quant/quantize_awq.py`](scripts/quant/quantize_awq.py) | REPORT C.1 · README §4 |
| **C. Serving** (vLLM, concurrent requests)                                | [`scripts/serving/serve.sh`](scripts/serving/serve.sh), [`concurrency_demo.py`](scripts/serving/concurrency_demo.py) | REPORT C.2–C.3 · README §3.5, §5 |
| **C. Resource Management** (isolate train/inference, shared GPU)          | `configs/serve.yaml` | REPORT C.4 · README §6 |
| **D. Automated Evaluation** (offline metrics)                             | [`scripts/eval/run_offline_eval.py`](scripts/eval/run_offline_eval.py), [`src/sales_agent/evals/rules.py`](src/sales_agent/evals/rules.py) | REPORT D.1 |
| **D. LLM-as-a-Judge** (Referee, coherence + sales logic)                  | [`scripts/eval/run_judge.py`](scripts/eval/run_judge.py), `configs/prompts/judge.j2` | REPORT D.2 |
| **Bonus 1** — 1,000+ concurrency, KV cache, PagedAttention/Spec-decoding  | [`Bonus/bonus1_capacity/`](Bonus/bonus1_capacity/) + `scripts/bench/` | REPORT Bonus 1 + C.3 |
| **Bonus 2** — advanced PTQ (NVFP4/ModelOpt) + quantization smoke test     | [`Bonus/bonus2_quant/`](Bonus/bonus2_quant/) | REPORT Bonus 2 + C.1 |
| **Bonus 3** — observability & benchmarking (Locust, Langfuse)             | [`Bonus/bonus3_observability/`](Bonus/bonus3_observability/) + `scripts/bench/` | REPORT Bonus 3 + C.3 |
| **Deliverable: Engineering README** (choices + run instructions)          | — | this README (§2 choices, §3 run, §4–§6) |
| **Deliverable: Performance Report** (loss curves + throughput vs latency) | — | REPORT B.1 (loss), C.3 (throughput vs latency) |
| **Deliverable: Bonus**                                                    | [`Bonus/`](Bonus/) | bonus 1–3 implemented as code + READMEs + committed results |
