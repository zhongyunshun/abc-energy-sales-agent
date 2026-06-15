"""Guard tests for the M8 serve stage.

Two jobs:
1. configs/serve.yaml encodes the design-doc M8 invariants, re-tuned for the
   4070 (gpu-mem-util lowered, compressed-tensors auto-detect, qwen3 reasoning
   parser). Paths asserted OS-independently via Path(...).as_posix().endswith.
2. docker/compose.yaml's serve command MIRRORS serve.yaml exactly -- serve.yaml
   is the single source of truth, compose is the literal copy that lets
   `docker compose up serve` run standalone, and this test forbids drift between
   the two (no scattered magic numbers).

No GPU, no Docker -- pure file parsing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sales_agent.common.config import load_config

REPO = Path(__file__).resolve().parents[1]
SERVE_CONFIG = REPO / "configs" / "serve.yaml"
COMPOSE = REPO / "docker" / "compose.yaml"


def _cfg() -> dict:
    return load_config(SERVE_CONFIG)


def posix(s: str) -> str:
    # load_config resolves *_dir keys to absolute OS-native paths; normalise to
    # forward slashes so the suffix check is OS-independent (Windows backslashes
    # otherwise break endswith).
    return Path(s).as_posix()


def _compose_serve() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["serve"]


def _parse_command(command: str) -> dict[str, object]:
    """Parse a vLLM `--flag value` / `--flag` (boolean) command string into a dict.

    A token starting with `--` whose following token is also a flag (or end of
    string) is a boolean flag (stored True); otherwise it consumes the next token
    as its value.
    """
    tokens = command.split()
    args: dict[str, object] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        assert tok.startswith("--"), f"unexpected non-flag token {tok!r} in command"
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[tok] = tokens[i + 1]
            i += 2
        else:
            args[tok] = True
            i += 1
    return args


# --- serve.yaml invariants -------------------------------------------------


def test_seed_default():
    assert _cfg()["seed"] == 42


def test_image_pinned():
    img = _cfg()["image"]
    assert img.startswith("vllm/vllm-openai:")
    assert img != "vllm/vllm-openai:latest"  # must be a pinned tag


def test_model_wired():
    m = _cfg()["model"]
    # AWQ INT4 product from M7 (host path resolved to absolute by load_config).
    assert posix(m["dir"]).endswith("models/quantized/awq")
    # In-container path is left literal (not rewritten to a host path on Windows).
    assert m["in_container"] == "/models/quantized/awq"
    assert m["served_name"]  # non-empty served model name for M9/M10/M11


def test_engine_4070_safe():
    e = _cfg()["engine"]
    # Lowered from the design-doc default 0.90 (the card also drives the display).
    assert 0.3 <= e["gpu_memory_utilization"] <= 0.7
    # Capped well below the model's 262144 ctx; voice dialogue needs little.
    assert 2048 <= e["max_model_len"] <= 4096
    assert 1 <= e["max_num_seqs"] <= 32
    assert e["enable_prefix_caching"] is True


def test_quantization_auto_detect():
    # MUST NOT force AutoAWQ; M7 output is compressed-tensors (vLLM auto-detects).
    assert _cfg()["engine"]["quantization"] is None


def test_reasoning_parser_qwen3():
    # Strips the empty <think></think> prefix into reasoning_content at the serve layer.
    assert _cfg()["engine"]["reasoning_parser"] == "qwen3"


def test_server_block():
    s = _cfg()["server"]
    assert s["health_endpoint"] == "/health"
    assert s["health_timeout_s"] == 120  # serve.sh exit 3 on timeout
    assert s["startup_poll_interval_s"] > 0
    assert 1 <= int(s["port"]) <= 65535


# --- compose.yaml mirrors serve.yaml (no drift) ----------------------------


def test_compose_image_matches():
    assert _compose_serve()["image"] == _cfg()["image"]


def test_compose_mounts_models():
    # ../models -> /models so /models/quantized/awq resolves inside the container.
    assert "../models:/models" in _compose_serve()["volumes"]


def test_compose_command_mirrors_serve_yaml():
    cfg = _cfg()
    m, e = cfg["model"], cfg["engine"]
    args = _parse_command(_compose_serve()["command"])

    assert args["--model"] == m["in_container"]
    assert args["--served-model-name"] == m["served_name"]
    assert args["--gpu-memory-utilization"] == str(e["gpu_memory_utilization"])
    assert args["--max-model-len"] == str(e["max_model_len"])
    assert args["--max-num-seqs"] == str(e["max_num_seqs"])
    assert args["--reasoning-parser"] == e["reasoning_parser"]
    # enable_prefix_caching true => the boolean flag is present.
    assert args.get("--enable-prefix-caching") is True
    # quantization null => the flag must be ABSENT (auto-detect compressed-tensors).
    assert "--quantization" not in args


def test_compose_port_published():
    cfg = _cfg()
    port = str(cfg["server"]["port"])
    published = _compose_serve()["ports"]
    assert any(f"{port}:{port}" == p for p in published)
