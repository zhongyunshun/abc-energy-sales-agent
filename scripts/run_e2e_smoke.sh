#!/usr/bin/env bash
# M12 / T12.1 -- end-to-end smoke test.
#
# Chains every module's --smoke entry ONCE, in an isolated workspace, to prove the
# M1->M11 pipeline stays connected in a clean environment. This is a CONNECTIVITY
# check, not a quality check: each stage runs with tiny inputs / few steps so the
# whole chain finishes in minutes. It is the Day-3 "reproducibility" gate referenced
# by the smoke workflow and backs the README "how to reproduce" section.
#
# What runs (decisions locked with the maintainer):
#   M1 normalize   (host, CPU)        -> $SMOKE/interim/normalized.jsonl
#   M2 synthesize  (OpenRouter API)   -> only with --with-api; default uses
#                                        tests/fixtures as stand-in M2 products
#   M3 split       (host, CPU)        -> $SMOKE/processed/{train,val,test}.jsonl
#   M4 SFT         (train container)  -> $SMOKE/adapters/sft  (4-bit, max_steps=10)
#   M5 DPO         -- SKIPPED on this 12GB 4070. dpo.yaml loads bf16 (load_in_4bit
#                     false) and needs >=24GB (validated on an A100); the card here
#                     would OOM. M6 merges the SFT adapter instead, so the
#                     M4->M6->M7 chain stays connected. DPO connectivity itself is
#                     covered by its own --smoke on a >=24GB host.
#   M6 merge       (train container)  -> $SMOKE/merged  (device_map=cpu; the 4070
#                     also drives the display, so a bf16 4B on GPU can OOM)
#   M7 quant AWQ   (train container)  -> $SMOKE/quantized/awq (32-sample calibration)
#   M8 serve       (vLLM container)   -> serves the REAL accepted models/quantized/awq.
#                     compose.yaml hard-codes that path, and a smoke-quantized model
#                     from a 10-step SFT is unusable; M7's own load+generate self-check
#                     already proves the quant->loadable chain. serve is stopped at the
#                     end (compose down), and on any failure via the EXIT trap.
#   demo/M9/M11    (host -> endpoint) concurrency demo (4 reqs), offline eval (10 rows),
#                     load test (one 30s tier) against the live endpoint.
#   M10 judge      (OpenRouter API)   -> only with --with-api, after M9 (5/group).
#
# Isolation: every artifact lands under reports/_smoke/<timestamp>/ so the real,
# accepted products in data/ models/ reports/ are never overwritten. (It must live
# under reports/ -- that is the only writable repo subtree mounted into the train
# container; /tmp is not visible inside it.) The few scripts whose INPUTS live only
# in their config (M3 inputs, M4 data paths, M7 calibration) get a patched COPY of
# the config written into the smoke dir. No module code or committed config is touched.
# The reports/_smoke/ tree is a throwaway -- delete it (or `git clean`) afterwards.
#
# Default = offline (no OpenRouter calls). Pass --with-api to exercise M2 + M10
# against the real API (requires OPENROUTER_API_KEY). M1 still loads its public HF
# dataset (cached locally after the first run); that is a data dependency, not the
# gated OpenRouter API.
#
# Exit codes: 0 = the whole chain is green. Any step's non-zero exit aborts
# immediately, prints which step failed, stops serve, and a per-step
# code/duration summary is printed at the end.
#
# Usage (from anywhere):
#   bash scripts/run_e2e_smoke.sh [--with-api]
set -uo pipefail

# --- args ---------------------------------------------------------------------
WITH_API=0
for arg in "$@"; do
  case "$arg" in
    --with-api) WITH_API=1 ;;
    -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

# --- resolve repo root from this script's location ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMPOSE_FILE="docker/compose.yaml"
# Relative smoke dir (under reports/, which is bind-mounted into the train
# container). Relative paths also dodge the space in the Windows repo path.
S="reports/_smoke/$(date +%Y%m%d_%H%M%S)"

# --- per-step runner + summary ------------------------------------------------
declare -a STEP_LOG=()
SUMMARY_PRINTED=0

print_summary() {
  [ "${SUMMARY_PRINTED}" -eq 1 ] && return 0
  SUMMARY_PRINTED=1
  echo ""
  echo "================ E2E smoke summary ================"
  echo "  smoke dir: ${S}   (with_api=${WITH_API})"
  local line
  for line in "${STEP_LOG[@]}"; do
    echo "  ${line}"
  done
  echo "==================================================="
}

run_step() {
  local name="$1"; shift
  echo ""
  echo ">>> [${name}] $*"
  local start; start=$(date +%s)
  "$@"
  local code=$?
  local dur=$(( $(date +%s) - start ))
  if [ "${code}" -eq 0 ]; then
    echo "<<< [${name}] OK (${dur}s)"
    STEP_LOG+=("OK    ${name}  (${dur}s)")
  else
    echo "!!! [${name}] FAILED (exit ${code}, after ${dur}s)" >&2
    STEP_LOG+=("FAIL  ${name}  (exit=${code}, ${dur}s)")
    print_summary
    exit "${code}"
  fi
}

teardown() {
  echo ""
  echo "[teardown] stopping serve (docker compose down) ..."
  docker compose -f "${COMPOSE_FILE}" down --remove-orphans >/dev/null 2>&1 || true
}
trap teardown EXIT

# Run a Python entrypoint inside the GPU train container. The Dockerfile ENTRYPOINT
# is /bin/bash, so the documented `run ... train uv run ...` becomes `bash uv ...`
# and fails; --entrypoint uv + starting at `run` is the working invocation
# (see memory: gpu-train-container-invocation).
train_run() {
  docker compose -f "${COMPOSE_FILE}" run --rm --entrypoint uv train run python "$@"
}

# --- step implementations -----------------------------------------------------
step_preflight() {
  command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; return 3; }
  docker info >/dev/null 2>&1 || { echo "ERROR: Docker is not running." >&2; return 3; }
  # The single 4070 is serial: the GPU must be free before the train stages.
  if docker ps --format '{{.Image}}' | grep -q '^sales-agent-train'; then
    echo "ERROR: a 'sales-agent-train' container is running -- free the GPU first." >&2
    return 3
  fi
  if [ "${WITH_API}" -eq 1 ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: --with-api set but OPENROUTER_API_KEY is empty." >&2
    return 3
  fi
  mkdir -p "${S}/interim" "${S}/configs"
  echo "smoke workspace: ${S}  | with_api=${WITH_API}"
}

# Write patched COPIES of the configs whose inputs/placement can't be set on the
# CLI, pointing them at the smoke workspace. No committed file is modified.
step_gen_configs() {
  SMOKE_DIR="${S}" uv run python - <<'PY'
import os
from pathlib import Path
import yaml

S = os.environ["SMOKE_DIR"]
out = Path(S) / "configs"
out.mkdir(parents=True, exist_ok=True)

def load(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))

def dump(cfg, name):
    (out / name).write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

# M1: a little more volume so the tiny stratified split keeps non-empty val/test.
n = load("configs/normalize.yaml")
n["smoke_limit"] = 120
dump(n, "normalize.yaml")

# M3: read the M1/M2 smoke products from the smoke dir, keep ALL M1 (ratio high so
# the downsample never starves the split), and write into the smoke dir.
s = load("configs/split.yaml")
s["inputs"] = [
    {"path": f"{S}/interim/normalized.jsonl", "role": "m1"},
    {"path": f"{S}/interim/synthetic_dialogues.jsonl", "role": "m2"},
]
s["downsample"]["ratio"] = 100
s["output_dir"] = f"{S}/processed"
dump(s, "split.yaml")

# M4 SFT: read the smoke split (the adapter dir comes from --output-dir). The loss
# curve + tensorboard dir are SEPARATE config keys (not under output_dir), so they
# must be redirected too -- otherwise the smoke run clobbers the real, committed
# reports/training/sft_loss.png.
sft = load("configs/sft.yaml")
sft["data"]["train_path"] = f"{S}/processed/train.jsonl"
sft["data"]["val_path"] = f"{S}/processed/val.jsonl"
sft["report"]["loss_curve_path"] = f"{S}/adapters/sft/sft_loss.png"
sft["logging"]["tensorboard_dir"] = f"{S}/tb"
dump(sft, "sft.yaml")

# M6 merge: load on CPU -- the 4070 also drives the display, so a bf16 4B on the
# GPU can OOM; smoke speed does not matter (adapter & output dir come from the CLI).
m = load("configs/merge.yaml")
m["model"]["device_map"] = "cpu"
dump(m, "merge.yaml")

# M7 quant: calibrate from the smoke train split (merged-dir & output via the CLI).
q = load("configs/quant.yaml")
q["calibration"]["source_path"] = f"{S}/processed/train.jsonl"
dump(q, "quant.yaml")

print("wrote patched configs ->", out)
PY
}

# M2 stand-in: real API with --with-api, else committed fixtures as M2 products.
step_m2() {
  if [ "${WITH_API}" -eq 1 ]; then
    uv run python scripts/data/synthesize.py --config configs/synthesize.yaml \
      --mode dialogues --smoke --output-dir "${S}/interim" || return $?
    uv run python scripts/data/synthesize.py --config configs/synthesize.yaml \
      --mode preferences --smoke --output-dir "${S}/interim" || return $?
  else
    echo "offline mode: using fixtures as stand-in M2 products (pass --with-api for real)"
    cp tests/fixtures/dialogues_valid.jsonl "${S}/interim/synthetic_dialogues.jsonl" || return $?
    cp tests/fixtures/preference_pairs_valid.jsonl "${S}/interim/preference_pairs.jsonl" || return $?
  fi
}

# =============================================================================
# main chain
# =============================================================================
run_step "preflight"      step_preflight
run_step "gen-configs"    step_gen_configs

# --- M1->M3: data pipeline (host, CPU) ---
run_step "M1 normalize"   uv run python scripts/data/normalize.py \
  --config "${S}/configs/normalize.yaml" --smoke --output-dir "${S}/interim"
run_step "M2 synth/fixt"  step_m2
run_step "M3 split"       uv run python scripts/data/split.py \
  --config "${S}/configs/split.yaml" --smoke --output-dir "${S}/processed"

# --- M4,M6,M7: train chain (GPU train container, serial; M5 DPO skipped) ---
run_step "M4 SFT"         train_run scripts/training/train_sft.py \
  --config "${S}/configs/sft.yaml" --smoke --output-dir "${S}/adapters/sft"
run_step "M6 merge"       train_run scripts/training/merge_adapter.py \
  --config "${S}/configs/merge.yaml" --adapter "${S}/adapters/sft" \
  --output-dir "${S}/merged" --smoke
run_step "M7 quant"       train_run scripts/quant/quantize_awq.py \
  --config "${S}/configs/quant.yaml" --merged-dir "${S}/merged" \
  --output-dir "${S}/quantized/awq" --smoke

# --- M8 serve (REAL AWQ product) + downstream clients ---
run_step "M8 serve"       bash scripts/serving/serve.sh
run_step "demo"           uv run python scripts/serving/concurrency_demo.py \
  --n 4 --output-dir "${S}/serving"
run_step "M9 eval"        uv run python scripts/eval/run_offline_eval.py \
  --config configs/eval_offline.yaml --model-tag smoke \
  --endpoint http://127.0.0.1:8000/v1 --smoke --output-dir "${S}/eval_offline"
if [ "${WITH_API}" -eq 1 ]; then
  run_step "M10 judge"    uv run python scripts/eval/run_judge.py \
    --config configs/eval_judge.yaml --inputs "${S}/eval_offline/smoke" \
    --smoke --output-dir "${S}/eval_judge"
fi
run_step "M11 bench"      uv run python scripts/bench/run_bench.py \
  --config configs/bench.yaml --host http://127.0.0.1:8000 \
  --smoke --output-dir "${S}/bench"

print_summary
echo ""
echo "E2E smoke PASSED. Throwaway artifacts under ${S} (safe to delete)."
