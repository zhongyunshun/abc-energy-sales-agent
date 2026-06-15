"""Guard tests for the M11 bench stage config (the M11 contract).

configs/bench.yaml is the single source of truth for the Locust benchmark; this
pins the design-doc invariants (concurrency ladder, per-tier duration, warm-up
discard, endpoint, generation params, seed=42). No GPU, no service -- pure file
parsing. Paths asserted OS-independently via Path(...).as_posix().endswith.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config

REPO = Path(__file__).resolve().parents[1]
BENCH_CONFIG = REPO / "configs" / "bench.yaml"


def _cfg() -> dict:
    return load_config(BENCH_CONFIG)


def posix(s: str) -> str:
    # load_config resolves *_path keys to absolute OS-native paths; normalise to
    # forward slashes so endswith is OS-independent (Windows backslashes break it).
    return Path(s).as_posix()


def test_seed_default():
    assert _cfg()["seed"] == 42


def test_test_path_wired():
    # M3 product; resolved to an absolute path by load_config.
    assert posix(_cfg()["test_path"]).endswith("data/processed/test.jsonl")


def test_output_dir_wired():
    assert posix(_cfg()["output_dir"]).endswith("reports/bench")


def test_endpoint_block():
    e = _cfg()["endpoint"]
    # host is the BASE (no /v1) -- left literal (not a *_path key).
    assert e["host"].startswith("http://") or e["host"].startswith("https://")
    assert not e["host"].rstrip("/").endswith("/v1")
    # URL routes must stay LITERAL (non-path key names) -- a "*_path" key would be
    # rewritten by load_config to an absolute host filesystem path and break the URL.
    assert e["chat_route"] == "/v1/chat/completions"
    assert e["served_model"]  # non-empty served model name (M8 --served-model-name)
    assert e["health_route"] == "/health"


def test_ladder_invariants():
    lad = _cfg()["ladder"]
    assert lad["concurrency"] == [1, 4, 8, 16, 32]  # design-doc ladder
    # 32 deliberately exceeds M8 max_num_seqs=16 to measure the queueing knee.
    assert max(lad["concurrency"]) > 16
    assert lad["run_time_s"] == 120
    assert lad["warmup_s"] == 15
    # Steady window must be positive after discarding warm-up.
    assert lad["run_time_s"] > lad["warmup_s"]


def test_generation_reproducible():
    g = _cfg()["generation"]
    assert g["temperature"] == 0.0  # greedy -> comparable with M8 demo / M9 eval
    assert g["max_tokens"] == 256


def test_smoke_block():
    s = _cfg()["smoke"]
    # One short tier to validate the pipeline before the full ladder.
    assert s["run_time_s"] < _cfg()["ladder"]["run_time_s"]
    assert s["warmup_s"] < s["run_time_s"]
    assert len(s["concurrency"]) >= 1
