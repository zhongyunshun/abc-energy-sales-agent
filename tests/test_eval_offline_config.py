"""Guard test for configs/eval_offline.yaml (the M9 contract invariants).

The config is the single source of truth for the offline-eval stage; this pins the
reproducibility invariants (greedy temperature=0, fixed max_tokens, the over_length
budget, seed=42) so no magic number drifts into code. Paths are asserted
OS-independently via Path(...).as_posix().endswith(...). Pure file parsing, no GPU.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config

REPO = Path(__file__).resolve().parents[1]
EVAL_CONFIG = REPO / "configs" / "eval_offline.yaml"


def _cfg() -> dict:
    return load_config(EVAL_CONFIG)


def posix(s: str) -> str:
    return Path(s).as_posix()


def test_seed_default():
    assert _cfg()["seed"] == 42


def test_test_path_wired():
    # load_config resolves *_path keys to absolute OS-native paths.
    assert posix(_cfg()["test_path"]).endswith("data/processed/test.jsonl")


def test_output_dir_wired():
    assert posix(_cfg()["output_dir"]).endswith("reports/eval_offline")


def test_generation_reproducible():
    g = _cfg()["generation"]
    assert g["temperature"] == 0.0  # greedy => reproducible across the three groups
    assert isinstance(g["max_tokens"], int) and g["max_tokens"] > 0
    assert g["concurrency"] >= 1


def test_over_length_threshold():
    # Voice replies are short (the M9 offline-eval target); the budget must be a positive int.
    thr = _cfg()["rules"]["over_length_max_tokens"]
    assert isinstance(thr, int) and thr > 0


def test_gathering_scenarios():
    assert "info_gathering" in _cfg()["rules"]["gathering_scenarios"]


def test_sampling_default_full_set():
    # null => full test set (the chosen default; M10 draws the same ids from it).
    assert _cfg()["sampling"]["n_samples"] is None


def test_smoke_block():
    assert _cfg()["smoke"]["n_samples"] >= 1


def test_endpoint_block():
    e = _cfg()["endpoint"]
    assert e["default"].endswith("/v1")
    assert e["served_model"]  # non-empty served model name
