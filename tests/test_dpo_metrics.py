"""Unit tests for the DPO reward/margin curve export (T5.5). Pure logic + a
headless render; no GPU. Feeds a synthetic DPO trainer_state and asserts series
extraction and that a real, non-empty PNG is produced.
"""

from __future__ import annotations

import json

import pytest

from sales_agent.training.dpo_metrics import extract_reward_metrics, plot_reward_margins

# Shaped like a TRL DPOTrainer log_history: each train step logs loss + rewards.
DPO_STATE = {
    "log_history": [
        {"loss": 0.69, "rewards/chosen": 0.01, "rewards/rejected": -0.01,
         "rewards/margins": 0.02, "rewards/accuracies": 0.5, "step": 1},
        {"loss": 0.60, "rewards/chosen": 0.05, "rewards/rejected": -0.08,
         "rewards/margins": 0.13, "rewards/accuracies": 0.7, "step": 2},
        {"loss": 0.52, "rewards/chosen": 0.09, "rewards/rejected": -0.20,
         "rewards/margins": 0.29, "rewards/accuracies": 0.85, "step": 3},
        {"train_runtime": 42.0, "step": 3},  # final summary entry, no rewards
    ]
}


def test_extract_reward_metrics_splits_series():
    series = extract_reward_metrics(DPO_STATE)
    assert series["rewards/margins"] == [(1, 0.02), (2, 0.13), (3, 0.29)]
    assert series["rewards/chosen"] == [(1, 0.01), (2, 0.05), (3, 0.09)]
    assert series["rewards/rejected"] == [(1, -0.01), (2, -0.08), (3, -0.20)]
    assert series["rewards/accuracies"] == [(1, 0.5), (2, 0.7), (3, 0.85)]


def test_extract_skips_entries_without_step():
    state = {"log_history": [{"rewards/margins": 9.9}, {"rewards/margins": 0.1, "step": 5}]}
    series = extract_reward_metrics(state)
    assert series["rewards/margins"] == [(5, 0.1)]


def test_plot_reward_margins_writes_png(tmp_path):
    out = tmp_path / "training" / "dpo_margins.png"
    result = plot_reward_margins(DPO_STATE, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_plot_reward_margins_accepts_json_path(tmp_path):
    state_path = tmp_path / "trainer_state.json"
    state_path.write_text(json.dumps(DPO_STATE), encoding="utf-8")
    out = tmp_path / "dpo_margins.png"
    plot_reward_margins(state_path, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_reward_margins_raises_without_margins():
    with pytest.raises(ValueError, match="no rewards/margins"):
        plot_reward_margins({"log_history": [{"loss": 1.0, "step": 1}]}, "unused.png")


def test_dpo_loss_curve_reuses_m4_plotter(tmp_path):
    # DPO loss curve reuses M4's plot_loss_curve unchanged (DPO logs `loss`).
    from sales_agent.training.plotting import plot_loss_curve

    out = tmp_path / "dpo_loss.png"
    plot_loss_curve(DPO_STATE, out, title="DPO training loss")
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
