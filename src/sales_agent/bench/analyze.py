"""Streaming-timing aggregation (design doc 3-M8 / 3-M11, task T8.3).

Pure functions that turn a stream of per-chunk arrival timestamps into latency
metrics. This is the M8 concurrency demo's measurement core AND the M11 (Locust)
benchmark's aggregation core, so it lives in ``src/`` and is unit-tested with no
network or GPU.

Conventions:
- All timestamps/durations are seconds (monotonic clock, e.g. ``time.perf_counter``).
- ``summarize_stream`` works on ONE request's chunk arrivals relative to its start.
- ``aggregate_streams`` / ``percentile`` summarise many requests for the demo table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class StreamStats:
    """Latency summary for one streamed response (all seconds, None if no chunks).

    - ``ttft_s``: time to first token = first chunk arrival - request start.
    - ``total_s``: end-to-end = last chunk arrival - request start.
    - ``itls_s``: inter-token latencies = gaps between consecutive chunk arrivals
      (empty for a 0- or 1-chunk stream).
    - ``itl_mean_s``: mean of ``itls_s`` (None when fewer than 2 chunks).
    """

    ttft_s: float | None
    total_s: float | None
    itls_s: list[float] = field(default_factory=list)
    itl_mean_s: float | None = None
    n_chunks: int = 0
    n_output_tokens: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_stream(
    start: float,
    chunk_times: Sequence[float],
    output_tokens: int | None = None,
) -> StreamStats:
    """Summarise one request's streamed chunk arrivals.

    ``start`` is the request-send timestamp; ``chunk_times`` are the arrival
    timestamps of each streamed chunk that carried content, in order. They must be
    non-decreasing and not precede ``start`` (otherwise the input is malformed and a
    ValueError is raised, rather than emitting negative latencies).

    Boundaries:
    - empty stream -> ttft/total/itl_mean = None, itls = [].
    - single chunk -> ttft = total = (chunk - start), itls = [], itl_mean = None.
    """
    n = len(chunk_times)
    if n == 0:
        return StreamStats(
            ttft_s=None, total_s=None, itls_s=[], itl_mean_s=None,
            n_chunks=0, n_output_tokens=output_tokens,
        )

    prev = start
    for i, t in enumerate(chunk_times):
        if t < prev:
            where = "before start" if i == 0 else f"before chunk {i - 1}"
            raise ValueError(f"chunk {i} timestamp {t} is {where} ({prev})")
        prev = t

    ttft_s = chunk_times[0] - start
    total_s = chunk_times[-1] - start
    itls_s = [chunk_times[i] - chunk_times[i - 1] for i in range(1, n)]
    itl_mean_s = (sum(itls_s) / len(itls_s)) if itls_s else None
    return StreamStats(
        ttft_s=ttft_s, total_s=total_s, itls_s=itls_s, itl_mean_s=itl_mean_s,
        n_chunks=n, n_output_tokens=output_tokens,
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); matches numpy's default.

    Raises ValueError on an empty sequence. q is clamped to [0, 100].
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    q = max(0.0, min(100.0, q))
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo))


@dataclass(frozen=True)
class AggregateStats:
    """Cross-request rollup for the concurrency demo summary table."""

    n_requests: int
    n_with_output: int
    ttft_mean_s: float | None
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    total_mean_s: float | None
    total_p50_s: float | None
    total_p95_s: float | None
    total_output_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


def aggregate_streams(stats: Sequence[StreamStats]) -> AggregateStats:
    """Roll up per-request :class:`StreamStats`; ignores streams with no chunks."""
    valid = [s for s in stats if s.ttft_s is not None]
    ttfts = [s.ttft_s for s in valid if s.ttft_s is not None]
    totals = [s.total_s for s in valid if s.total_s is not None]
    tokens = sum(s.n_output_tokens or 0 for s in valid)

    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    def _pct(xs: list[float], q: float) -> float | None:
        return percentile(xs, q) if xs else None

    return AggregateStats(
        n_requests=len(stats),
        n_with_output=len(valid),
        ttft_mean_s=_mean(ttfts),
        ttft_p50_s=_pct(ttfts, 50),
        ttft_p95_s=_pct(ttfts, 95),
        total_mean_s=_mean(totals),
        total_p50_s=_pct(totals, 50),
        total_p95_s=_pct(totals, 95),
        total_output_tokens=tokens,
    )


def batching_speedup(single_latency_s: float, concurrent_wall_s: float, n: int) -> float:
    """Continuous-batching speedup: (n serial requests) / (n concurrent requests).

    ``n * single_latency_s`` estimates running n requests back-to-back; dividing by
    the measured concurrent wall-clock shows how much continuous batching saved.
    A value >> 1 is the evidence that batching is active. Raises ValueError on a
    non-positive wall time.
    """
    if concurrent_wall_s <= 0:
        raise ValueError("concurrent_wall_s must be positive")
    return (n * single_latency_s) / concurrent_wall_s


def throughput_tok_s(total_output_tokens: int, wall_s: float) -> float:
    """Aggregate output throughput (tokens/second). Raises ValueError on wall<=0."""
    if wall_s <= 0:
        raise ValueError("wall_s must be positive")
    return total_output_tokens / wall_s
