"""M11 bench aggregation + plotting (the M11 contract, task T11.3).

Pure logic that turns the per-tier raw CSVs (one row per request, written by the
locustfile) into the design-doc 2.4 ``bench_summary.csv`` and the three report
PNGs. The latency math reuses the already-tested M8 core
(``analyze.percentile`` / ``analyze.throughput_tok_s``) -- this module only adds
warm-up truncation and the per-tier rollup. matplotlib is imported lazily with the
headless Agg backend (no display, cheap import) like training/plotting.py.

Definitions (steady state = after discarding the first ``warmup_s`` of a tier):
- steady window = ``run_time_s - warmup_s`` seconds.
- throughput_tok_s = sum(output tokens of steady successful requests) / window.
- req_s = goodput = steady SUCCESSFUL requests / window.
- error_rate = steady failed requests / steady total requests.
- ttft_p50/p95 = percentiles of steady successful requests' TTFT (ms).
- itl_p50/p95 = percentiles of steady successful requests' mean ITL (ms).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sales_agent.bench.analyze import percentile, throughput_tok_s
from sales_agent.bench.locust_logic import RAW_CSV_COLUMNS

# Output filename committed under reports/bench/ for M12.
SUMMARY_CSV_NAME = "bench_summary.csv"

# design-doc 2.4 contract column order for bench_summary.csv.
BENCH_SUMMARY_COLUMNS = [
    "concurrency",
    "ttft_p50",
    "ttft_p95",
    "itl_p50",
    "itl_p95",
    "throughput_tok_s",
    "req_s",
    "error_rate",
]


@dataclass(frozen=True)
class TierSummary:
    """One concurrency tier's steady-state rollup (the §2.4 row + provenance)."""

    concurrency: int
    ttft_p50: float | None
    ttft_p95: float | None
    itl_p50: float | None
    itl_p95: float | None
    throughput_tok_s: float
    req_s: float
    error_rate: float
    # provenance (not in the §2.4 CSV, kept for the manifest / report):
    n_total: int = 0
    n_ok: int = 0
    n_failed: int = 0
    steady_window_s: float = 0.0

    def as_summary_row(self) -> dict:
        """The §2.4 columns only, rounded for the CSV."""

        def r(x: float | None, nd: int) -> float | str:
            return "" if x is None else round(x, nd)

        return {
            "concurrency": self.concurrency,
            "ttft_p50": r(self.ttft_p50, 2),
            "ttft_p95": r(self.ttft_p95, 2),
            "itl_p50": r(self.itl_p50, 3),
            "itl_p95": r(self.itl_p95, 3),
            "throughput_tok_s": r(self.throughput_tok_s, 2),
            "req_s": r(self.req_s, 3),
            "error_rate": r(self.error_rate, 4),
        }

    def as_dict(self) -> dict:
        return asdict(self)


def _to_float(s: str) -> float | None:
    s = (s or "").strip()
    return None if s == "" else float(s)


def _to_int(s: str) -> int | None:
    s = (s or "").strip()
    return None if s == "" else int(float(s))


def read_raw_csv(path: str | Path) -> list[dict]:
    """Read a locustfile raw CSV into typed rows (start_ts/ms floats, ok bool)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(RAW_CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: raw CSV missing columns {sorted(missing)}")
        for r in reader:
            rows.append(
                {
                    "start_ts": float(r["start_ts"]),
                    "ttft_ms": _to_float(r["ttft_ms"]),
                    "itl_mean_ms": _to_float(r["itl_mean_ms"]),
                    "total_ms": _to_float(r["total_ms"]),
                    "n_output_tokens": _to_int(r["n_output_tokens"]),
                    "ok": str(r["ok"]).strip().lower() == "true",
                }
            )
    return rows


def truncate_warmup(rows: Sequence[dict], warmup_s: float) -> list[dict]:
    """Drop requests that STARTED within the first ``warmup_s`` of the tier.

    The boundary is relative to the earliest request's ``start_ts`` so it is robust
    to when the run actually began. ``warmup_s <= 0`` keeps everything.
    """
    if not rows or warmup_s <= 0:
        return list(rows)
    t0 = min(r["start_ts"] for r in rows)
    boundary = t0 + warmup_s
    return [r for r in rows if r["start_ts"] >= boundary]


def summarize_tier(
    rows: Sequence[dict], concurrency: int, run_time_s: float, warmup_s: float
) -> TierSummary:
    """Roll up one tier's raw rows into a :class:`TierSummary` (steady state only)."""
    steady_window_s = run_time_s - warmup_s
    if steady_window_s <= 0:
        raise ValueError(f"steady window {steady_window_s}s must be positive (run>{warmup_s})")

    steady = truncate_warmup(rows, warmup_s)
    n_total = len(steady)
    ok_rows = [r for r in steady if r["ok"]]
    n_ok = len(ok_rows)
    n_failed = n_total - n_ok

    ttfts = [r["ttft_ms"] for r in ok_rows if r["ttft_ms"] is not None]
    itls = [r["itl_mean_ms"] for r in ok_rows if r["itl_mean_ms"] is not None]
    tokens = sum(r["n_output_tokens"] or 0 for r in ok_rows)

    def _p(xs: list[float], q: float) -> float | None:
        return percentile(xs, q) if xs else None

    return TierSummary(
        concurrency=concurrency,
        ttft_p50=_p(ttfts, 50),
        ttft_p95=_p(ttfts, 95),
        itl_p50=_p(itls, 50),
        itl_p95=_p(itls, 95),
        throughput_tok_s=throughput_tok_s(tokens, steady_window_s),
        req_s=n_ok / steady_window_s,
        error_rate=(n_failed / n_total) if n_total else 0.0,
        n_total=n_total,
        n_ok=n_ok,
        n_failed=n_failed,
        steady_window_s=steady_window_s,
    )


def summarize_from_raw(
    raw_path: str | Path, concurrency: int, run_time_s: float, warmup_s: float
) -> TierSummary:
    return summarize_tier(read_raw_csv(raw_path), concurrency, run_time_s, warmup_s)


def raw_csv_path(output_dir: str | Path, concurrency: int) -> Path:
    return Path(output_dir) / f"raw_{concurrency}.csv"


def collect_summaries(
    output_dir: str | Path, tiers: Sequence[int], run_time_s: float, warmup_s: float
) -> list[TierSummary]:
    """Summarise every tier whose raw_{c}.csv exists, ordered by concurrency."""
    out: list[TierSummary] = []
    for c in sorted(tiers):
        p = raw_csv_path(output_dir, c)
        if p.exists():
            out.append(summarize_from_raw(p, c, run_time_s, warmup_s))
    return out


def write_summary_csv(summaries: Sequence[TierSummary], path: str | Path) -> Path:
    """Write bench_summary.csv with the exact §2.4 column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_SUMMARY_COLUMNS)
        writer.writeheader()
        for s in sorted(summaries, key=lambda t: t.concurrency):
            writer.writerow(s.as_summary_row())
    return path


# --- plotting (headless Agg) ------------------------------------------------


def _line_plot(
    out_path: str | Path,
    xs: Sequence[int],
    series: Sequence[tuple[str, list[float | None], str]],
    title: str,
    ylabel: str,
):
    """Plot one or more (label, y-values-aligned-to-xs, color) series vs concurrency.

    ``None`` y-values are dropped per series (a tier where the metric is undefined,
    e.g. all requests failed) so a missing point never renders as zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys, color in series:
        pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
        if not pts:
            continue
        ax.plot(
            [x for x, _ in pts], [y for _, y in pts],
            label=label, color=color, marker="o", markersize=4,
        )
    ax.set_xlabel("concurrency (in-flight requests)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(xs))
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_throughput(summaries: Sequence[TierSummary], out_path: str | Path) -> Path:
    s = sorted(summaries, key=lambda t: t.concurrency)
    xs = [t.concurrency for t in s]
    return _line_plot(
        out_path, xs,
        [("output tok/s", [t.throughput_tok_s for t in s], "#1f77b4")],
        "Throughput vs concurrency", "output tokens / s",
    )


def plot_ttft(summaries: Sequence[TierSummary], out_path: str | Path) -> Path:
    s = sorted(summaries, key=lambda t: t.concurrency)
    xs = [t.concurrency for t in s]
    return _line_plot(
        out_path, xs,
        [
            ("TTFT p50", [t.ttft_p50 for t in s], "#1f77b4"),
            ("TTFT p95", [t.ttft_p95 for t in s], "#d62728"),
        ],
        "Time to first token vs concurrency", "TTFT (ms)",
    )


def plot_itl(summaries: Sequence[TierSummary], out_path: str | Path) -> Path:
    s = sorted(summaries, key=lambda t: t.concurrency)
    xs = [t.concurrency for t in s]
    return _line_plot(
        out_path, xs,
        [
            ("ITL p50", [t.itl_p50 for t in s], "#1f77b4"),
            ("ITL p95", [t.itl_p95 for t in s], "#d62728"),
        ],
        "Inter-token latency vs concurrency", "ITL mean (ms)",
    )


# PNG filenames committed under reports/bench/ for M12.
PLOT_FILES = {
    "throughput": "throughput_vs_concurrency.png",
    "ttft": "ttft_vs_concurrency.png",
    "itl": "itl_vs_concurrency.png",
}


def write_plots(summaries: Sequence[TierSummary], output_dir: str | Path) -> dict[str, Path]:
    """Write all three report PNGs into ``output_dir``; returns {name: path}."""
    output_dir = Path(output_dir)
    return {
        "throughput": plot_throughput(summaries, output_dir / PLOT_FILES["throughput"]),
        "ttft": plot_ttft(summaries, output_dir / PLOT_FILES["ttft"]),
        "itl": plot_itl(summaries, output_dir / PLOT_FILES["itl"]),
    }
