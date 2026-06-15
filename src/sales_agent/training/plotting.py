"""Loss-curve export from a HuggingFace ``trainer_state.json`` (the M4/M5 contract).

The PNG and the underlying ``trainer_state.json`` are core source material for
the M12 README performance section (the M12 report contract), so this must be
real, reproducible output. The data extraction is a pure function (unit-tested);
``matplotlib`` is imported lazily with the headless ``Agg`` backend so plotting
works in the container with no display and so importing this module stays cheap.
"""

from __future__ import annotations

import json
from pathlib import Path


def extract_loss_history(
    trainer_state: dict,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Split a trainer ``log_history`` into train and eval ``(step, loss)`` series.

    HF Trainer logs interleaved dicts: training entries carry ``loss`` (+``step``)
    and evaluation entries carry ``eval_loss`` (+``step``). Entries lacking a
    usable step are skipped. Returns ``(train_series, eval_series)``, each sorted
    by step.
    """
    log = trainer_state.get("log_history", [])
    train: list[tuple[int, float]] = []
    eval_: list[tuple[int, float]] = []
    for entry in log:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train.append((int(step), float(entry["loss"])))
        if "eval_loss" in entry:
            eval_.append((int(step), float(entry["eval_loss"])))
    train.sort()
    eval_.sort()
    return train, eval_


def plot_loss_curve(
    trainer_state: dict | str | Path,
    out_path: str | Path,
    title: str = "SFT training loss",
) -> Path:
    """Plot train/eval loss vs. step to ``out_path`` (PNG); returns the path.

    ``trainer_state`` may be the parsed dict or a path to ``trainer_state.json``.
    Raises ``ValueError`` if there is no training-loss data to plot (so a silent
    empty figure never masquerades as a real loss curve).
    """
    if isinstance(trainer_state, (str, Path)):
        trainer_state = json.loads(Path(trainer_state).read_text(encoding="utf-8"))

    train, eval_ = extract_loss_history(trainer_state)
    if not train:
        raise ValueError("trainer_state has no training-loss entries to plot")

    import matplotlib

    matplotlib.use("Agg")  # headless: no display in the container
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([s for s, _ in train], [v for _, v in train], label="train", color="#1f77b4")
    if eval_:
        ax.plot(
            [s for s, _ in eval_],
            [v for _, v in eval_],
            label="eval",
            color="#d62728",
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
