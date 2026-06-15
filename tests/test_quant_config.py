"""Guard tests for configs/quant.yaml: the design-doc M7 invariants are encoded
faithfully -- W4A16 asymmetric AWQ at group_size 128, training-domain calibration
(256 full / 32 smoke), the merged input and AWQ output wiring, and committed
report paths. No GPU.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quant.yaml"


def _cfg() -> dict:
    return load_config(CONFIG)


def posix(s: str) -> str:
    # load_config resolves *_path / *_dir keys to absolute OS-native paths (design
    # doc 1.4); normalise to forward slashes before the suffix check so the assertion
    # is OS-independent (a backslash separator breaks endswith on Windows).
    return Path(s).as_posix()


def test_seed_default():
    assert _cfg()["seed"] == 42


def test_recipe_is_w4a16_asym_group128():
    r = _cfg()["recipe"]
    assert r["scheme"] == "W4A16_ASYM"  # asymmetric, canonical AWQ (user decision)
    assert r["group_size"] == 128
    assert r["symmetric"] is False
    assert r["targets"] == ["Linear"]
    assert r["ignore"] == ["lm_head"]
    assert r["duo_scaling"] == "both"


def test_calibration_sizes():
    c = _cfg()["calibration"]
    assert c["n_samples"] == 256
    assert c["smoke_n_samples"] == 32
    assert 512 <= c["max_seq_len"] <= 1024  # design-doc calibration window
    assert c["batch_size"] == 1
    assert c["pipeline"] == "sequential"
    assert posix(c["source_path"]).endswith("data/processed/train.jsonl")


def test_model_and_output_wired():
    cfg = _cfg()
    # Merged FP16 input (the M6 product on this PC) + AWQ INT4 output.
    assert posix(cfg["model"]["merged_dir"]).endswith("models/adapters/merged")
    assert posix(cfg["output_dir"]).endswith("models/quantized/awq")
    assert cfg["model"]["dtype"] == "bfloat16"
    # CPU load so the sequential pipeline offloads per block (4070-safe).
    assert cfg["model"]["device_map"] in ("cpu", "auto")


def test_self_check_block():
    s = _cfg()["self_check"]
    assert s["prompt"].strip()
    assert s["max_new_tokens"] > 0


def test_probe_block():
    p = _cfg()["probe"]
    assert isinstance(p["enabled"], bool)
    assert posix(p["prompts_path"]).endswith("tests/fixtures/quant_probe_prompts.jsonl")
    assert p["max_new_tokens"] > 0


def test_committed_report_paths_wired():
    r = _cfg()["report"]
    assert posix(r["manifest_path"]).endswith("reports/training/quant_manifest.json")
    assert posix(r["size_report_path"]).endswith("reports/training/quant_report.md")
