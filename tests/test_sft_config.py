"""Guard tests for configs/sft.yaml: the design-doc baseline table is encoded
faithfully and the masking markers stay in sync with formatting.py constants.

These catch silent drift (a magic number sneaking into code, or a marker change
in one place but not the other) without needing a GPU.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.training.formatting import INSTRUCTION_PART, RESPONSE_PART

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sft.yaml"


def _cfg() -> dict:
    return load_config(CONFIG)


def test_baseline_hyperparameters_match_design_table():
    cfg = _cfg()
    assert cfg["seed"] == 42
    assert cfg["model"]["max_seq_length"] == 2048
    assert cfg["model"]["load_in_4bit"] is True

    lora = cfg["lora"]
    # r32/alpha64 chosen from the HP sweep (was r16/alpha32 baseline).
    assert (lora["r"], lora["alpha"]) == (32, 64)
    assert lora["alpha"] == 2 * lora["r"]  # alpha = 2*rank heuristic
    assert lora["dropout"] == 0.0
    assert lora["use_gradient_checkpointing"] == "unsloth"
    assert lora["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]

    tr = cfg["train"]
    assert tr["num_train_epochs"] == 2
    assert tr["per_device_train_batch_size"] * tr["gradient_accumulation_steps"] == 16
    assert tr["learning_rate"] == 2e-4
    assert tr["lr_scheduler_type"] == "cosine"
    assert tr["warmup_ratio"] == 0.05
    assert tr["optim"] == "adamw_8bit"


def test_masking_markers_match_formatting_constants():
    # The config feeds these strings to Unsloth train_on_responses_only; they
    # MUST equal the formatting.py constants or masking silently diverges.
    cfg = _cfg()
    assert cfg["masking"]["strategy"] == "train_on_responses_only"
    assert cfg["masking"]["instruction_part"] == INSTRUCTION_PART
    assert cfg["masking"]["response_part"] == RESPONSE_PART


def test_smoke_block_present():
    cfg = _cfg()
    assert cfg["smoke"]["max_steps"] == 10
    assert cfg["smoke"]["n_samples"] == 64
