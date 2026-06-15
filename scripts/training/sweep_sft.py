"""M4 hyperparameter sweep: vary LoRA rank/alpha and learning rate over a small
grid at a FIXED compute budget, compare validation loss, emit a comparison table.

This is the empirical evidence behind the "choice of hyperparameters" deliverable
(rubric: LoRA rank, alpha, learning rate). The grid is intentionally small -- a
3-day challenge wants a justified choice validated, not an exhaustive search.

Each combo runs as a fresh subprocess of ``train_sft.py`` (clean GPU state) on a
temp config derived from a base config, for a fractional epoch so the whole sweep
finishes in ~1h on an A100. torch.compile is disabled for the sweep (it is
orthogonal to the HP comparison and recompiling each run wastes time). The
pure-logic helpers (:func:`expand_grid`, :func:`render_table`) are unit-tested;
orchestration is GPU and runs in the train venv/container.

Usage (in the GPU train env):
    uv run python scripts/training/sweep_sft.py --base-config configs/sft_a100.yaml \
        --epochs 0.4 --output-root models/adapters/sweep \
        --report reports/training/sft_hparam_sweep.md
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from sales_agent.common.config import find_repo_root
from sales_agent.training.plotting import extract_loss_history

logger = logging.getLogger("sweep_sft")


def expand_grid(ranks: list[int], lrs: list[float]) -> list[dict]:
    """Cartesian product of (rank, lr) with the alpha = 2*rank heuristic."""
    return [
        {"r": r, "alpha": 2 * r, "lr": lr} for r, lr in itertools.product(ranks, lrs)
    ]


def best_eval_loss(trainer_state_path: str | Path) -> float | None:
    """Lowest eval_loss recorded in a trainer_state.json (None if no eval logged)."""
    state = json.loads(Path(trainer_state_path).read_text(encoding="utf-8"))
    _, evals = extract_loss_history(state)
    return min((v for _, v in evals), default=None)


def render_table(results: list[dict]) -> str:
    """Markdown comparison table, sorted by best val loss (best first)."""
    header = (
        "| LoRA r | alpha | lr | steps | best val loss | status |\n"
        "|---|---|---|---|---|---|"
    )
    rows = []
    for x in sorted(
        results, key=lambda d: (d["best_eval"] if d["best_eval"] is not None else float("inf"))
    ):
        be = f"{x['best_eval']:.4f}" if x.get("best_eval") is not None else "n/a"
        status = "ok" if x.get("rc") == 0 else f"FAILED(rc={x.get('rc')})"
        rows.append(
            f"| {x['r']} | {x['alpha']} | {x['lr']:.0e} | {x.get('steps', '?')} | {be} | {status} |"
        )
    return header + "\n" + "\n".join(rows)


def _run_one(base: dict, combo: dict, epochs: float, output_dir: Path, train_script: Path) -> dict:
    """Write a temp config with the combo's overrides, run train_sft.py, collect loss."""
    cfg = copy.deepcopy(base)
    cfg["lora"]["r"] = combo["r"]
    cfg["lora"]["alpha"] = combo["alpha"]
    cfg["train"]["learning_rate"] = combo["lr"]
    cfg["train"]["num_train_epochs"] = epochs
    cfg["train"]["torch_compile"] = False  # orthogonal to HP; faster sweep
    cfg["train"]["save_strategy"] = "no"  # no intermediate checkpoints during sweep

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = output_dir / "config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    logger.info(
        "running combo r=%d alpha=%d lr=%.0e -> %s",
        combo["r"], combo["alpha"], combo["lr"], output_dir,
    )
    proc = subprocess.run(
        [sys.executable, str(train_script),
         "--config", str(tmp_cfg), "--output-dir", str(output_dir)]
    )
    state_path = output_dir / "trainer_state.json"
    be = best_eval_loss(state_path) if proc.returncode == 0 and state_path.exists() else None
    steps = None
    if state_path.exists():
        steps = json.loads(state_path.read_text(encoding="utf-8")).get("global_step")
    return {**combo, "rc": proc.returncode, "best_eval": be, "steps": steps}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True)
    p.add_argument("--epochs", type=float, default=0.4, help="fractional epochs per combo")
    p.add_argument("--ranks", default="16,32", help="comma-separated LoRA ranks")
    p.add_argument("--lrs", default="2e-4,1e-4", help="comma-separated learning rates")
    p.add_argument("--output-root", default="models/adapters/sweep")
    p.add_argument("--report", default="reports/training/sft_hparam_sweep.md")
    args = p.parse_args(argv)

    root = find_repo_root()
    train_script = root / "scripts" / "training" / "train_sft.py"
    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    ranks = [int(x) for x in args.ranks.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    combos = expand_grid(ranks, lrs)
    output_root = Path(args.output_root)
    logger.info("sweep: %d combos x %.2f epochs", len(combos), args.epochs)

    results = []
    for combo in combos:
        tag = f"r{combo['r']}_lr{combo['lr']:.0e}"
        results.append(_run_one(base, combo, args.epochs, output_root / tag, train_script))

    table = render_table(results)
    note = (
        f"# M4 SFT hyperparameter sweep\n\n"
        f"Base config: `{args.base_config}` | budget: {args.epochs} epoch(s)/combo | "
        f"alpha = 2*rank | torch.compile off (HP-orthogonal).\n\n"
        f"{table}\n"
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(note, encoding="utf-8")
    logger.info("wrote sweep report -> %s\n%s", report_path, table)
    return 0 if all(r["rc"] == 0 for r in results) else 3


if __name__ == "__main__":
    sys.exit(main())
