"""Unit tests for the M11 bench aggregation + plotting (task T11.3).

Pins warm-up truncation, the per-tier rollup numbers (cross-checked against
analyze.percentile / throughput_tok_s), the §2.4 CSV column order, raw-CSV
round-trip, and that the three PNGs render. No GPU, no service.
"""

from __future__ import annotations

import csv

import pytest

from sales_agent.bench.analyze import percentile, throughput_tok_s
from sales_agent.bench.locust_logic import RAW_CSV_COLUMNS, assemble_row
from sales_agent.bench.report import (
    BENCH_SUMMARY_COLUMNS,
    TierSummary,
    collect_summaries,
    raw_csv_path,
    read_raw_csv,
    summarize_tier,
    truncate_warmup,
    write_plots,
    write_summary_csv,
)


def _row(start_ts, ttft_ms, itl_ms, total_ms, tokens, ok):
    return {
        "start_ts": start_ts,
        "ttft_ms": ttft_ms,
        "itl_mean_ms": itl_ms,
        "total_ms": total_ms,
        "n_output_tokens": tokens,
        "ok": ok,
    }


# --- truncate_warmup -------------------------------------------------------


def test_truncate_warmup_drops_by_start_time():
    rows = [
        _row(100.0, 10, 1, 50, 5, True),   # t0 -> in warmup, dropped
        _row(105.0, 10, 1, 50, 5, True),   # before boundary (t0+10=110), dropped
        _row(110.0, 10, 1, 50, 5, True),   # at boundary -> kept
        _row(120.0, 10, 1, 50, 5, True),   # kept
    ]
    steady = truncate_warmup(rows, warmup_s=10.0)
    assert [r["start_ts"] for r in steady] == [110.0, 120.0]


def test_truncate_warmup_zero_keeps_all():
    rows = [_row(0.0, 1, 1, 1, 1, True), _row(1.0, 1, 1, 1, 1, True)]
    assert truncate_warmup(rows, 0) == rows
    assert truncate_warmup([], 5) == []


# --- summarize_tier numbers (cross-checked with analyze.py) -----------------


def test_summarize_tier_matches_analyze():
    # t0=0, warmup 10s -> only start_ts>=10 are steady. Steady window = 100-10 = 90s.
    rows = [
        _row(0.0, 999, 99, 999, 999, True),    # warmup -> excluded entirely
        _row(10.0, 100.0, 20.0, 400.0, 30, True),
        _row(20.0, 200.0, 40.0, 800.0, 50, True),
        _row(30.0, 300.0, 60.0, 1200.0, 70, True),
        _row(40.0, None, None, None, None, False),  # steady failure
    ]
    s = summarize_tier(rows, concurrency=8, run_time_s=100.0, warmup_s=10.0)

    assert s.n_total == 4 and s.n_ok == 3 and s.n_failed == 1
    assert s.steady_window_s == 90.0

    ttfts = [100.0, 200.0, 300.0]
    itls = [20.0, 40.0, 60.0]
    assert s.ttft_p50 == pytest.approx(percentile(ttfts, 50))
    assert s.ttft_p95 == pytest.approx(percentile(ttfts, 95))
    assert s.itl_p50 == pytest.approx(percentile(itls, 50))
    assert s.itl_p95 == pytest.approx(percentile(itls, 95))
    # throughput counts only steady SUCCESSFUL tokens (30+50+70) over the window.
    assert s.throughput_tok_s == pytest.approx(throughput_tok_s(150, 90.0))
    # req_s = goodput = steady successful requests / window.
    assert s.req_s == pytest.approx(3 / 90.0)
    assert s.error_rate == pytest.approx(1 / 4)


def test_summarize_tier_all_failed_gives_none_percentiles():
    rows = [_row(10.0, None, None, None, None, False), _row(20.0, None, None, None, None, False)]
    s = summarize_tier(rows, concurrency=32, run_time_s=100.0, warmup_s=5.0)
    assert s.ttft_p50 is None and s.ttft_p95 is None
    assert s.itl_p50 is None and s.itl_p95 is None
    assert s.throughput_tok_s == 0.0
    assert s.req_s == 0.0
    assert s.error_rate == 1.0


def test_summarize_tier_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="steady window"):
        summarize_tier([_row(0.0, 1, 1, 1, 1, True)], 1, run_time_s=10.0, warmup_s=10.0)


# --- raw CSV round-trip (locustfile row -> read_raw_csv) --------------------


def test_raw_csv_roundtrip(tmp_path):
    # Write rows exactly like the locustfile does (assemble_row + DictWriter), read back.
    written = [
        assemble_row(1000.0, [1000.5, 1000.7, 1000.9], output_tokens=3, ok=True),
        assemble_row(1001.0, [], output_tokens=None, ok=False),  # failure -> None fields
    ]
    p = tmp_path / "raw_4.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_CSV_COLUMNS)
        w.writeheader()
        w.writerows(written)

    rows = read_raw_csv(p)
    assert len(rows) == 2
    assert rows[0]["ok"] is True
    assert rows[0]["ttft_ms"] == pytest.approx(500.0, abs=1e-6)
    assert rows[0]["n_output_tokens"] == 3
    assert rows[1]["ok"] is False
    assert rows[1]["ttft_ms"] is None and rows[1]["n_output_tokens"] is None


def test_read_raw_csv_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("start_ts,ttft_ms\n1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_raw_csv(p)


# --- bench_summary.csv contract --------------------------------------------


def _summary(c, **kw):
    base = dict(
        ttft_p50=10.0, ttft_p95=20.0, itl_p50=1.0, itl_p95=2.0,
        throughput_tok_s=100.0, req_s=1.0, error_rate=0.0,
    )
    base.update(kw)
    return TierSummary(concurrency=c, **base)


def test_summary_csv_has_contract_columns_in_order(tmp_path):
    p = tmp_path / "bench_summary.csv"
    write_summary_csv([_summary(16), _summary(1), _summary(4)], p)
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == BENCH_SUMMARY_COLUMNS  # exact §2.4 order
    # rows sorted by concurrency ascending.
    assert [r[0] for r in body] == ["1", "4", "16"]


def test_summary_csv_writes_blank_for_none_percentiles(tmp_path):
    p = tmp_path / "bench_summary.csv"
    write_summary_csv([_summary(32, ttft_p50=None, ttft_p95=None)], p)
    with open(p, encoding="utf-8", newline="") as f:
        row = list(csv.DictReader(f))[0]
    assert row["ttft_p50"] == "" and row["ttft_p95"] == ""
    assert row["throughput_tok_s"] == "100.0"


# --- collect_summaries + plotting ------------------------------------------


def test_collect_summaries_reads_existing_tiers(tmp_path):
    for c in (1, 4):
        rows = [
            assemble_row(float(t), [float(t) + 0.1], output_tokens=2, ok=True) for t in range(20)
        ]
        p = raw_csv_path(tmp_path, c)
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    summaries = collect_summaries(tmp_path, tiers=[1, 4, 8], run_time_s=30.0, warmup_s=5.0)
    assert [s.concurrency for s in summaries] == [1, 4]  # tier 8 has no raw CSV -> skipped


def test_write_plots_produces_three_pngs(tmp_path):
    summaries = [_summary(1), _summary(4), _summary(16, ttft_p50=None, ttft_p95=None)]
    paths = write_plots(summaries, tmp_path)
    assert set(paths) == {"throughput", "ttft", "itl"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
