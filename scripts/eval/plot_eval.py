"""Plot the M9/M10 evaluation results as figures for the performance report.

Reads the committed evaluation artifacts and renders three grouped-bar charts
(no new data, just a visualization of what M9/M10 already produced):

  reports/eval_offline/rule_trigger_rates.png   <- M9 summary.json (per group)
  reports/eval_offline/reply_length.png         <- M9 summary.json (per group)
  reports/eval_judge/judge_scores.png           <- M10 aggregate.json (2 judges avg)

Run:
    uv run python scripts/eval/plot_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Larger fonts so the charts stay legible when scaled down inside slide columns.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

REPO_ROOT = Path(__file__).resolve().parents[2]

GROUPS = ["base", "sft", "dpo"]
GROUP_LABELS = {"base": "base", "sft": "SFT", "dpo": "SFT+DPO"}
# Consistent palette across every figure (matches the deck's accent blue).
GROUP_COLORS = {"base": "#b0b0b0", "sft": "#005AA0", "dpo": "#2ca02c"}


def _bars(ax, categories, series, fmt, ylabel, title, ymax=None):
    """Grouped bar chart: one bar group per category, one bar per model group."""
    n = len(GROUPS)
    width = 0.8 / n
    x = range(len(categories))
    for i, g in enumerate(GROUPS):
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, series[g], width, label=GROUP_LABELS[g], color=GROUP_COLORS[g])
        for b, v in zip(bars, series[g]):
            ax.annotate(
                fmt(v),
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9, xytext=(0, 1),
                textcoords="offset points",
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_offline(out_dir: Path):
    summaries = {}
    for g in GROUPS:
        p = out_dir / g / "summary.json"
        summaries[g] = json.loads(p.read_text())["overall"]
    return summaries


def plot_rule_rates(summaries, out_path: Path):
    rules = ["made_up_price", "over_length", "role_break", "no_question_in_gathering"]
    labels = ["made-up\nprice", "over-length\n(>120 tok)", "role\nbreak", "no question\n(info)"]
    series = {g: [summaries[g]["rule_rates"][r] * 100 for r in rules] for g in GROUPS}
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    _bars(ax, labels, series, lambda v: f"{v:.1f}", "trigger rate (%)",
          "M9 rule trigger rate by model (lower is better)", ymax=80)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reply_length(summaries, out_path: Path):
    stats = ["mean", "p50", "p90", "p95"]
    labels = ["mean", "p50", "p90", "p95"]
    series = {g: [summaries[g]["length_tokens"][s] for s in stats] for g in GROUPS}
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    _bars(ax, labels, series, lambda v: f"{v:.0f}", "reply length (tokens)",
          "M9 reply length by model (voice budget = 120 tok)")
    ax.axhline(120, ls="--", lw=1.2, color="#c0392b")
    ax.annotate("120-tok budget", (3.3, 124), color="#c0392b", fontsize=10, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_judge_scores(aggregate_path: Path, out_path: Path):
    agg = json.loads(aggregate_path.read_text())
    dims = ["coherence", "sales_logic", "professionalism", "hallucination"]
    labels = ["coherence", "sales\nlogic", "professional-\nism", "halluc.-\nfree"]
    judges = agg["judges"]
    # average the two judges' overall means per (group, dimension)
    series = {}
    for g in GROUPS:
        series[g] = [
            sum(judges[j]["overall"][g][d]["mean"] for j in judges) / len(judges)
            for d in dims
        ]
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    _bars(ax, labels, series, lambda v: f"{v:.2f}", "score (1-5, higher is better)",
          "M10 LLM-judge score by model (avg of 2 judges)", ymax=5.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-dir", default=str(REPO_ROOT / "reports" / "eval_offline"))
    ap.add_argument("--judge-dir", default=str(REPO_ROOT / "reports" / "eval_judge"))
    args = ap.parse_args()

    offline_dir = Path(args.offline_dir)
    judge_dir = Path(args.judge_dir)

    summaries = _load_offline(offline_dir)
    plot_rule_rates(summaries, offline_dir / "rule_trigger_rates.png")
    plot_reply_length(summaries, offline_dir / "reply_length.png")
    plot_judge_scores(judge_dir / "aggregate.json", judge_dir / "judge_scores.png")

    print("wrote:")
    for p in [
        offline_dir / "rule_trigger_rates.png",
        offline_dir / "reply_length.png",
        judge_dir / "judge_scores.png",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
