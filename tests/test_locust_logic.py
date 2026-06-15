"""Unit tests for the pure logic behind the M11 locustfile (task T11.1).

Covers the SSE parsing the locustfile does on every chunk, the per-request timing
assembly (first chunk = TTFT, gaps = ITL, in milliseconds), and the prompt-pool
construction (context selection + length-bucket round-robin). No network.
"""

from __future__ import annotations

import pytest

from sales_agent.bench.locust_logic import (
    RAW_CSV_COLUMNS,
    assemble_row,
    order_by_length_buckets,
    parse_sse_chunk,
    select_context,
)

APPROX = 1e-9


# --- parse_sse_chunk -------------------------------------------------------


def test_parse_content_delta():
    line = 'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}'
    c = parse_sse_chunk(line)
    assert c.is_data and not c.done
    assert c.content == "Hello"
    assert c.completion_tokens is None


def test_parse_usage_chunk_has_tokens_no_content():
    # vLLM with include_usage emits a final choices=[] chunk carrying usage.
    line = 'data: {"choices":[],"usage":{"completion_tokens":42}}'
    c = parse_sse_chunk(line)
    assert c.is_data and not c.done
    assert c.content is None
    assert c.completion_tokens == 42


def test_parse_done_sentinel():
    c = parse_sse_chunk("data: [DONE]")
    assert c.is_data and c.done
    assert c.content is None


def test_parse_blank_and_keepalive_are_not_data():
    assert parse_sse_chunk("").is_data is False
    assert parse_sse_chunk("   ").is_data is False
    assert parse_sse_chunk(": ping").is_data is False  # SSE comment / keepalive


def test_parse_empty_delta_content_is_empty_string():
    # An empty delta (no visible token) -> "", which the caller treats as no chunk.
    line = 'data: {"choices":[{"delta":{}}]}'
    assert parse_sse_chunk(line).content is None
    line2 = 'data: {"choices":[{"delta":{"content":""}}]}'
    assert parse_sse_chunk(line2).content == ""


def test_parse_malformed_payload_raises():
    with pytest.raises(ValueError, match="malformed SSE"):
        parse_sse_chunk("data: {not json")


# --- assemble_row: per-request timing (ms) ---------------------------------


def test_assemble_row_ttft_and_itl_in_ms():
    # start=10.0; chunks at 10.5, 10.7, 10.9 -> TTFT 500ms, ITL mean 200ms,
    # total 900ms (delegates boundaries to summarize_stream).
    row = assemble_row(10.0, [10.5, 10.7, 10.9], output_tokens=3, ok=True)
    assert row["start_ts"] == 10.0
    assert row["ttft_ms"] == pytest.approx(500.0, abs=1e-6)
    assert row["itl_mean_ms"] == pytest.approx(200.0, abs=1e-6)
    assert row["total_ms"] == pytest.approx(900.0, abs=1e-6)
    assert row["n_output_tokens"] == 3
    assert row["ok"] is True


def test_assemble_row_single_chunk_has_no_itl():
    row = assemble_row(0.0, [0.4], output_tokens=1, ok=True)
    assert row["ttft_ms"] == pytest.approx(400.0, abs=1e-6)
    assert row["total_ms"] == pytest.approx(400.0, abs=1e-6)
    assert row["itl_mean_ms"] is None  # <2 chunks -> no inter-token latency


def test_assemble_row_failed_no_chunks():
    row = assemble_row(5.0, [], output_tokens=None, ok=False)
    assert row["ttft_ms"] is None
    assert row["total_ms"] is None
    assert row["itl_mean_ms"] is None
    assert row["ok"] is False


def test_raw_csv_columns_match_row_keys():
    row = assemble_row(0.0, [0.1, 0.2], output_tokens=2, ok=True)
    assert set(RAW_CSV_COLUMNS) == set(row.keys())


# --- select_context --------------------------------------------------------


def test_select_context_drops_trailing_assistant():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},  # dropped
    ]
    ctx = select_context(msgs)
    assert ctx is not None
    assert ctx[-1] == {"role": "user", "content": "u2"}
    assert all(set(m) == {"role", "content"} for m in ctx)


def test_select_context_rejects_non_user_ending():
    # After dropping a trailing assistant, ends with system -> unusable.
    msgs = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]
    assert select_context(msgs) is None
    assert select_context([]) is None


# --- order_by_length_buckets ------------------------------------------------


def _mk(i: int, length: int) -> tuple[str, list[dict]]:
    return (f"id-{i}", [{"role": "user", "content": "x" * length}])


def test_round_robin_rotates_lengths():
    # Lengths 1..9 -> sorted buckets [1,2,3],[4,5,6],[7,8,9]; round-robin picks
    # one per bucket: short, mid, long, short, mid, long, ...
    items = [_mk(i, length) for i, length in enumerate([5, 1, 9, 3, 7, 2, 8, 4, 6])]
    out = order_by_length_buckets(items, n_buckets=3)
    lengths = [len(m[0]["content"]) for _, m in out]
    assert lengths == [1, 4, 7, 2, 5, 8, 3, 6, 9]


def test_order_is_deterministic():
    items = [_mk(i, length) for i, length in enumerate([5, 1, 9, 3, 7, 2, 8, 4, 6])]
    assert order_by_length_buckets(items, 3) == order_by_length_buckets(items, 3)


def test_order_preserves_all_items():
    items = [_mk(i, length) for i, length in enumerate([5, 1, 9, 3, 7, 2, 8])]
    out = order_by_length_buckets(items, 3)
    assert sorted(i for i, _ in out) == sorted(i for i, _ in items)


def test_order_trivial_inputs():
    assert order_by_length_buckets([], 3) == []
    one = [_mk(0, 5)]
    assert order_by_length_buckets(one, 3) == one
