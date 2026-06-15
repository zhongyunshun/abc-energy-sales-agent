"""Unit tests for the loss-curve export (T4.4). Pure logic + a headless render;
no GPU. Feeds a synthetic trainer_state and asserts series extraction and that a
real, non-empty PNG is produced.
"""

from __future__ import annotations

import json

import pytest

from sales_agent.training.plotting import extract_loss_history, plot_loss_curve

TRAINER_STATE = {
    "log_history": [
        {"loss": 2.0, "step": 10, "epoch": 0.1},
        {"loss": 1.5, "step": 20, "epoch": 0.2},
        {"eval_loss": 1.6, "step": 20, "epoch": 0.2},
        {"loss": 1.2, "step": 30, "epoch": 0.3},
        {"eval_loss": 1.3, "step": 40, "epoch": 0.4},
        {"train_runtime": 123.4, "step": 40},  # final summary entry, no loss
    ]
}


def test_extract_loss_history_splits_series():
    train, eval_ = extract_loss_history(TRAINER_STATE)
    assert train == [(10, 2.0), (20, 1.5), (30, 1.2)]
    assert eval_ == [(20, 1.6), (40, 1.3)]


def test_extract_skips_entries_without_step():
    state = {"log_history": [{"loss": 9.9}, {"loss": 1.0, "step": 5}]}
    train, _ = extract_loss_history(state)
    assert train == [(5, 1.0)]


def test_plot_loss_curve_writes_png(tmp_path):
    out = tmp_path / "training" / "sft_loss.png"
    result = plot_loss_curve(TRAINER_STATE, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_plot_loss_curve_accepts_json_path(tmp_path):
    state_path = tmp_path / "trainer_state.json"
    state_path.write_text(json.dumps(TRAINER_STATE), encoding="utf-8")
    out = tmp_path / "sft_loss.png"
    plot_loss_curve(state_path, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_loss_curve_raises_without_train_loss():
    # An eval-only / empty log must NOT silently produce a blank curve.
    with pytest.raises(ValueError, match="no training-loss"):
        plot_loss_curve({"log_history": [{"eval_loss": 1.0, "step": 1}]}, "unused.png")
