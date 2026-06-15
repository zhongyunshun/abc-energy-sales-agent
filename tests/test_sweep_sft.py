"""Unit tests for the M4 hyperparameter-sweep pure-logic helpers (host, no GPU).

The orchestration in sweep_sft.main is GPU (subprocesses train_sft.py); only the
grid expansion, best-eval extraction, and table rendering are unit-tested here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "sweep_sft.py"


def _load():
    spec = importlib.util.spec_from_file_location("m4_sweep_sft", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expand_grid_alpha_is_twice_rank():
    sweep = _load()
    grid = sweep.expand_grid([16, 32], [2e-4, 1e-4])
    assert len(grid) == 4
    assert all(g["alpha"] == 2 * g["r"] for g in grid)
    assert {(g["r"], g["lr"]) for g in grid} == {
        (16, 2e-4), (16, 1e-4), (32, 2e-4), (32, 1e-4)
    }


def test_best_eval_loss_picks_minimum(tmp_path):
    sweep = _load()
    state = tmp_path / "trainer_state.json"
    state.write_text(
        '{"log_history": [{"eval_loss": 1.6, "step": 50}, {"eval_loss": 1.4, "step": 100},'
        ' {"eval_loss": 1.5, "step": 150}, {"loss": 2.0, "step": 100}]}',
        encoding="utf-8",
    )
    assert sweep.best_eval_loss(state) == 1.4


def test_render_table_sorts_best_first_and_marks_failures():
    sweep = _load()
    results = [
        {"r": 16, "alpha": 32, "lr": 2e-4, "steps": 294, "best_eval": 1.40, "rc": 0},
        {"r": 32, "alpha": 64, "lr": 1e-4, "steps": 294, "best_eval": 1.55, "rc": 0},
        {"r": 32, "alpha": 64, "lr": 2e-4, "steps": None, "best_eval": None, "rc": 1},
    ]
    table = sweep.render_table(results)
    lines = table.splitlines()
    # Header + separator + 3 rows.
    assert len(lines) == 5
    # Best val loss (1.40) row comes first among data rows; failure row last.
    assert "1.4000" in lines[2]
    assert "FAILED(rc=1)" in lines[-1]
    assert "n/a" in lines[-1]
