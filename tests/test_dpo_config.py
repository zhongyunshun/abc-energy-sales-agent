"""Guard tests for configs/dpo.yaml: the design-doc M5 knobs are encoded
faithfully (beta 0.1, lr 5e-6, 1 epoch, batch as SFT) and the bf16 / ref-scheme
invariants that keep the run compatible with the SFT adapter hold. No GPU.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dpo.yaml"


def _cfg() -> dict:
    return load_config(CONFIG)


def test_dpo_hyperparameters_match_design():
    cfg = _cfg()
    assert cfg["seed"] == 42
    d = cfg["dpo"]
    assert d["beta"] == 0.1
    assert d["learning_rate"] == 5e-6
    assert d["num_train_epochs"] == 1
    assert d["loss_type"] == "sigmoid"
    # effective batch = SFT's 16
    assert d["per_device_train_batch_size"] * d["gradient_accumulation_steps"] == 16


def test_load_precision_matches_sft_adapter():
    # The SFT adapter is bf16/r32 (load_in_4bit false). Loading 4-bit here would
    # misalign the adapter -> this MUST stay false.
    cfg = _cfg()
    assert cfg["model"]["load_in_4bit"] is False
    assert cfg["model"]["max_seq_length"] == 2048
    assert cfg["dpo"]["bf16"] is True


def test_reference_scheme_two_adapter_ref_sft():
    cfg = _cfg()
    ref = cfg["ref"]
    assert ref["scheme"] == "two_adapter"
    # policy named "default" -> saves to output_dir root; distinct frozen reference.
    assert ref["policy_adapter_name"] == "default"
    assert ref["ref_adapter_name"] == "reference"
    assert ref["policy_adapter_name"] != ref["ref_adapter_name"]


def test_inputs_and_artifacts_wired():
    cfg = _cfg()
    # load_config resolves *_path / *_dir keys to absolute OS-native paths (design
    # doc section 1.4), so normalise to forward slashes before the suffix check --
    # otherwise these assertions are brittle on Windows (backslash separators).
    def posix(s: str) -> str:
        return Path(s).as_posix()

    assert posix(cfg["sft_adapter"]).endswith("models/adapters/sft")
    assert posix(cfg["data"]["pref_path"]).endswith("data/interim/preference_pairs.jsonl")
    assert posix(cfg["probes"]["path"]).endswith("tests/fixtures/dpo_probes.jsonl")
    assert cfg["probes"]["max_new_tokens"] > 0
    assert posix(cfg["report"]["loss_curve_path"]).endswith("dpo_loss.png")
    assert posix(cfg["report"]["margins_curve_path"]).endswith("dpo_margins.png")
    assert posix(cfg["report"]["behavior_diff_path"]).endswith("dpo_behavior_diff.md")
    assert posix(cfg["dpo"]["output_dir"]).endswith("models/adapters/dpo")


def test_smoke_block_present():
    cfg = _cfg()
    assert cfg["smoke"]["max_steps"] == 10
    assert cfg["smoke"]["n_pairs"] == 32
