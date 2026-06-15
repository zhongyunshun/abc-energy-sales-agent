"""DPO reward/margin curve export from a ``trainer_state.json`` (T5.5).

TRL's ``DPOTrainer`` logs, per step, ``loss`` plus reward diagnostics
(``rewards/chosen``, ``rewards/rejected``, ``rewards/margins``,
``rewards/accuracies``). The *margins* curve (chosen-minus-rejected reward) is the
headline DPO signal -- a rising margin means the policy increasingly prefers the
chosen (professional / grounded) response over the rejected (pushy / hallucinated)
one. This module extracts those series and renders the curve; the DPO *loss* curve
reuses M4's :func:`plotting.plot_loss_curve` unchanged (DPO also logs ``loss``).

Pure data extraction (unit-tested) + a lazy headless ``Agg`` render, mirroring
``plotting.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

# TRL DPO log keys that carry a per-step scalar we plot.
REWARD_KEYS = ("rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies")


def extract_reward_metrics(trainer_state: dict) -> dict[str, list[tuple[int, float]]]:
    """Split a trainer ``log_history`` into per-key ``(step, value)`` series.

    Returns a dict keyed by each of :data:`REWARD_KEYS` present in the log, each
    value a step-sorted list of ``(step, value)``. Entries without a usable step
    are skipped. Missing keys map to empty lists.
    """
    log = trainer_state.get("log_history", [])
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in REWARD_KEYS}
    for entry in log:
        step = entry.get("step")
        if step is None:
            continue
        for key in REWARD_KEYS:
            if key in entry and entry[key] is not None:
                series[key].append((int(step), float(entry[key])))
    for key in series:
        series[key].sort()
    return series


def plot_reward_margins(
    trainer_state: dict | str | Path,
    out_path: str | Path,
    title: str = "DPO reward margins",
) -> Path:
    """Plot reward margins (+ chosen/rejected rewards) vs. step to ``out_path``.

    ``trainer_state`` may be the parsed dict or a path to ``trainer_state.json``.
    Raises ``ValueError`` if there is no ``rewards/margins`` data, so an empty
    figure never masquerades as a real DPO curve.
    """
    if isinstance(trainer_state, (str, Path)):
        trainer_state = json.loads(Path(trainer_state).read_text(encoding="utf-8"))

    series = extract_reward_metrics(trainer_state)
    margins = series["rewards/margins"]
    if not margins:
        raise ValueError("trainer_state has no rewards/margins entries to plot")

    import matplotlib

    matplotlib.use("Agg")  # headless: no display in the container
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([s for s, _ in margins], [v for _, v in margins], label="margin (chosen−rejected)",
            color="#2ca02c", marker="o", markersize=3)
    for key, color in (("rewards/chosen", "#1f77b4"), ("rewards/rejected", "#d62728")):
        pts = series[key]
        if pts:
            ax.plot([s for s, _ in pts], [v for _, v in pts], label=key.split("/")[-1],
                    color=color, alpha=0.6, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
