"""Unit tests for the M8 streaming-timing core (src/sales_agent/bench/analyze.py).

This is the M8 concurrency demo's measurement core and M11's aggregation core, so
the boundaries are pinned: first chunk = TTFT, chunk gaps = ITL, empty stream, and
single-chunk stream.
"""

from __future__ import annotations

import math

import pytest

from sales_agent.bench.analyze import (
    AggregateStats,
    StreamStats,
    aggregate_streams,
    batching_speedup,
    percentile,
    summarize_stream,
    throughput_tok_s,
)

APPROX = 1e-9


# --- summarize_stream: core boundaries -------------------------------------


def test_first_chunk_is_ttft():
    # start=10.0, first chunk at 10.5 -> TTFT = 0.5 (independent of later chunks).
    s = summarize_stream(10.0, [10.5, 10.7, 10.9])
    assert s.ttft_s == pytest.approx(0.5, abs=APPROX)
    assert s.n_chunks == 3


def test_chunk_gaps_are_itl():
    # gaps 0.2 and 0.3 -> itls=[0.2,0.3], mean 0.25; total = last-start = 0.6.
    s = summarize_stream(0.0, [0.1, 0.3, 0.6])
    assert s.itls_s == pytest.approx([0.2, 0.3], abs=1e-9)
    assert s.itl_mean_s == pytest.approx(0.25, abs=APPROX)
    assert s.total_s == pytest.approx(0.6, abs=APPROX)
    assert s.ttft_s == pytest.approx(0.1, abs=APPROX)


def test_empty_stream():
    s = summarize_stream(5.0, [])
    assert s == StreamStats(
        ttft_s=None, total_s=None, itls_s=[], itl_mean_s=None,
        n_chunks=0, n_output_tokens=None,
    )


def test_single_chunk():
    # total collapses to TTFT; no inter-token latencies.
    s = summarize_stream(2.0, [2.4])
    assert s.ttft_s == pytest.approx(0.4, abs=APPROX)
    assert s.total_s == pytest.approx(0.4, abs=APPROX)
    assert s.itls_s == []
    assert s.itl_mean_s is None
    assert s.n_chunks == 1


def test_output_tokens_passthrough():
    s = summarize_stream(0.0, [0.1, 0.2], output_tokens=7)
    assert s.n_output_tokens == 7


def test_chunk_before_start_raises():
    with pytest.raises(ValueError, match="before start"):
        summarize_stream(1.0, [0.9])


def test_non_monotonic_chunks_raise():
    with pytest.raises(ValueError, match="before chunk"):
        summarize_stream(0.0, [0.1, 0.3, 0.2])


def test_as_dict_roundtrip():
    d = summarize_stream(0.0, [0.5, 1.0], output_tokens=2).as_dict()
    assert d["ttft_s"] == pytest.approx(0.5, abs=APPROX)
    assert d["n_output_tokens"] == 2


# --- percentile ------------------------------------------------------------


def test_percentile_basic():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 50) == pytest.approx(2.5, abs=APPROX)  # linear interp


def test_percentile_single_value():
    assert percentile([7.0], 95) == 7.0


def test_percentile_clamps_and_rejects_empty():
    assert percentile([1.0, 2.0], 250) == 2.0  # clamped to 100
    with pytest.raises(ValueError):
        percentile([], 50)


# --- aggregate_streams -----------------------------------------------------


def test_aggregate_ignores_empty_streams():
    stats = [
        summarize_stream(0.0, [0.1, 0.5], output_tokens=2),  # ttft 0.1, total 0.5
        summarize_stream(0.0, [0.3, 0.9], output_tokens=4),  # ttft 0.3, total 0.9
        summarize_stream(0.0, []),                           # no output -> ignored
    ]
    agg = aggregate_streams(stats)
    assert isinstance(agg, AggregateStats)
    assert agg.n_requests == 3
    assert agg.n_with_output == 2
    assert agg.ttft_mean_s == pytest.approx(0.2, abs=APPROX)
    assert agg.total_mean_s == pytest.approx(0.7, abs=APPROX)
    assert agg.total_output_tokens == 6


def test_aggregate_all_empty():
    agg = aggregate_streams([summarize_stream(0.0, []), summarize_stream(0.0, [])])
    assert agg.n_with_output == 0
    assert agg.ttft_mean_s is None
    assert agg.ttft_p95_s is None
    assert agg.total_output_tokens == 0


# --- batching_speedup / throughput -----------------------------------------


def test_batching_speedup():
    # 16 requests x 1.0s serial = 16s; if concurrent wall is 2.0s -> 8x speedup.
    assert batching_speedup(1.0, 2.0, 16) == pytest.approx(8.0, abs=APPROX)


def test_batching_speedup_rejects_nonpositive_wall():
    with pytest.raises(ValueError):
        batching_speedup(1.0, 0.0, 16)


def test_throughput():
    assert throughput_tok_s(500, 2.0) == pytest.approx(250.0, abs=APPROX)
    with pytest.raises(ValueError):
        throughput_tok_s(500, 0.0)


def test_itl_mean_matches_manual():
    # extra guard: itl_mean equals statistics mean of the gaps.
    chunks = [0.0, 0.05, 0.2, 0.21, 0.5]
    s = summarize_stream(0.0, chunks)
    gaps = [chunks[i] - chunks[i - 1] for i in range(1, len(chunks))]
    assert s.itl_mean_s == pytest.approx(sum(gaps) / len(gaps), abs=APPROX)
    assert not math.isnan(s.itl_mean_s)
