"""Guard tests for configs/merge.yaml: the design-doc M6 invariants are encoded
faithfully -- bf16 (not 4-bit) load, the DPO adapter as merge source, the merged
output directory, and the consistency-check knobs. No GPU.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "merge.yaml"


def _cfg() -> dict:
    return load_config(CONFIG)


def posix(s: str) -> str:
    # load_config resolves *_path / *_dir keys to absolute OS-native paths (design
    # doc section 1.4); normalise to forward slashes before the suffix check so the
    # assertion is OS-independent (a backslash separator breaks endswith on Windows).
    return Path(s).as_posix()


def test_seed_default():
    assert _cfg()["seed"] == 42


def test_load_precision_is_bf16_not_4bit():
    # Merge in bf16 to match the adapter's training precision; 4-bit merging would
    # lose precision (design doc section 3-M6). There is no load_in_4bit knob here.
    cfg = _cfg()
    assert cfg["model"]["dtype"] == "bfloat16"
    assert cfg["model"]["name"] == "unsloth/Qwen3-4B-Instruct-2507"


def test_cpu_fallback_path_present():
    # device_map must exist so a 12GB card can fall back to "cpu" (design doc 3-M6).
    cfg = _cfg()
    assert "device_map" in cfg["model"]
    assert cfg["model"]["device_map"] in ("auto", "cpu", "cuda")


def test_adapter_and_output_wired():
    cfg = _cfg()
    # Source = the M5 DPO policy adapter; output = the dense merged model dir.
    assert posix(cfg["adapter"]).endswith("models/adapters/dpo")
    assert posix(cfg["output_dir"]).endswith("models/merged")


def test_consistency_block():
    c = _cfg()["consistency"]
    assert posix(c["prompts_path"]).endswith("tests/fixtures/merge_consistency_prompts.jsonl")
    assert c["match_mode"] in ("exact", "prefix_tokens")
    assert c["prefix_n"] == 64
    assert c["max_new_tokens"] > 0


def test_committed_report_paths_wired():
    # Full runs write auditable evidence under reports/training/ (committed), since
    # the merged model dir is gitignored.
    r = _cfg()["report"]
    assert posix(r["manifest_path"]).endswith("reports/training/merge_manifest.json")
    assert posix(r["consistency_report_path"]).endswith("reports/training/merge_consistency.md")


def test_smoke_block_present():
    s = _cfg()["smoke"]
    assert s["n_prompts"] >= 1
    assert s["max_new_tokens"] >= 1
