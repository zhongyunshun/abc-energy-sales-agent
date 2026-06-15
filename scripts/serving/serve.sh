#!/usr/bin/env bash
# M8 serve launcher (design doc 3-M8 / task T8.2).
#
# Wraps `docker compose up serve` with a pre-flight check and a /health poll.
# Reads ALL runtime values from configs/serve.yaml (the single source of truth)
# via load_config, so this script carries no magic numbers. Runs on the Windows
# host under Git Bash or WSL.
#
# Exit codes (design doc 1.4):
#   0  service healthy
#   2  input-contract failure (AWQ model dir missing)
#   3  external-dependency failure (Docker unavailable, or /health not ready
#      within server.health_timeout_s)
#
# Usage (from anywhere):
#   bash scripts/serving/serve.sh
set -euo pipefail

EXIT_CONTRACT=2
EXIT_DEP=3

# Resolve repo root from this script's location (scripts/serving/serve.sh) so the
# script works regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

COMPOSE_FILE="docker/compose.yaml"

# --- read serve.yaml once (load_config = repo-relative path resolution + seed) ---
# Print shell assignments and eval them. MODEL_DIR_EXISTS is computed in Python to
# avoid Windows-path issues with bash's `[ -d ]` on a "C:\..." string.
# String values are single-quoted (and MODEL_DIR uses forward slashes) so `eval`
# survives a Windows host path with spaces / backslashes (e.g. "...\Python Projects\...").
eval "$(uv run python - <<'PY'
from pathlib import Path
from sales_agent.common.config import load_config

c = load_config("configs/serve.yaml")
s, m = c["server"], c["model"]
print(f"SERVE_HOST='{s['host']}'")
print(f"SERVE_PORT={s['port']}")
print(f"HEALTH_ENDPOINT='{s['health_endpoint']}'")
print(f"HEALTH_TIMEOUT_S={s['health_timeout_s']}")
print(f"POLL_INTERVAL_S={s['startup_poll_interval_s']}")
print(f"MODEL_DIR_EXISTS={1 if Path(m['dir']).is_dir() else 0}")
print(f"MODEL_DIR='{Path(m['dir']).as_posix()}'")
PY
)"

# --- pre-flight: the AWQ product must exist (input contract) ---
if [ "${MODEL_DIR_EXISTS}" != "1" ]; then
  echo "ERROR: AWQ model dir not found: ${MODEL_DIR}" >&2
  echo "       Run M7 (quantize_awq.py) first, or fix model.dir in configs/serve.yaml." >&2
  exit "${EXIT_CONTRACT}"
fi

# --- pre-flight: Docker must be reachable (external dependency) ---
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not available (is Docker Desktop running?)." >&2
  exit "${EXIT_DEP}"
fi

# GPU is serial on the single 4070: warn (don't fail) if the train container is up.
if docker ps --format '{{.Image}}' | grep -q '^sales-agent-train'; then
  echo "WARNING: a 'sales-agent-train' container is running -- it shares the 4070." >&2
  echo "         Stop it before serving to avoid GPU contention (proposal section 7)." >&2
fi

# --- serve-compat shim: strip quant-config keys the pinned vLLM rejects ---
# M7's compressed-tensors 0.16.0 product carries scale_dtype/zp_dtype, which
# vLLM v0.10.2 forbids; this idempotent patch drops them so the model loads.
echo "Patching AWQ config.json for vLLM v0.10.2 compatibility ..."
uv run python scripts/serving/patch_quant_config.py --config configs/serve.yaml

# --- start the service (detached) ---
echo "Starting vLLM serve via ${COMPOSE_FILE} ..."
docker compose -f "${COMPOSE_FILE}" up -d serve

# --- poll /health until ready or timeout ---
HEALTH_URL="http://127.0.0.1:${SERVE_PORT}${HEALTH_ENDPOINT}"
echo "Waiting for ${HEALTH_URL} (timeout ${HEALTH_TIMEOUT_S}s) ..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
while true; do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "OK: vLLM is healthy at http://127.0.0.1:${SERVE_PORT}/v1"
    echo "    served model: query GET /v1/models for the name."
    exit 0
  fi
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "ERROR: /health not ready within ${HEALTH_TIMEOUT_S}s." >&2
    echo "       Recent logs:" >&2
    docker compose -f "${COMPOSE_FILE}" logs --tail 40 serve >&2 || true
    exit "${EXIT_DEP}"
  fi
  sleep "${POLL_INTERVAL_S}"
done
