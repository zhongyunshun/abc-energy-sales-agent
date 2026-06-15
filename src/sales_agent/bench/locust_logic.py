"""Pure logic behind the M11 Locust file (the M11 contract, task T11.1).

The locustfile itself (scripts/bench/locustfile.py) is a thin shell that drives
real HTTP streaming and writes a raw CSV row per request; it is NOT unit-tested
(real service). Everything it *computes* lives here so it can be unit-tested with
no network:

- ``parse_sse_chunk`` turns one streamed Server-Sent-Events line from vLLM's
  OpenAI-compatible ``/v1/chat/completions`` into (content piece, usage tokens,
  done) -- the parsing the locustfile does on every chunk.
- ``assemble_row`` is the per-request timing: it delegates the boundaries
  (first chunk = TTFT, gaps between chunks = ITL) to the already-tested
  ``analyze.summarize_stream`` and converts to the milliseconds the raw CSV
  stores. This is the "per-chunk timing logic" the task asks to unit-test.
- ``select_context`` / ``order_by_length_buckets`` build the prompt pool: each
  test dialogue's context up to the final assistant turn (same as M9 / the M8
  demo), bucketed by length and round-robined so every tier mixes lengths.

All seconds in, milliseconds out (the raw CSV and Locust both speak ms).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from sales_agent.bench.analyze import summarize_stream

# Raw per-request CSV columns (written by the locustfile, read back by report.py).
RAW_CSV_COLUMNS = ["start_ts", "ttft_ms", "itl_mean_ms", "total_ms", "n_output_tokens", "ok"]

_DATA_PREFIX = "data:"
_DONE_PAYLOAD = "[DONE]"


@dataclass(frozen=True)
class SSEChunk:
    """One parsed SSE line from the streamed chat completion.

    - ``is_data``: False for blank / comment (``:`` keepalive) lines.
    - ``done``: True for the terminal ``data: [DONE]`` line.
    - ``content``: the delta text (may be ``""`` for an empty delta, ``None`` when
      the chunk carries no choice, e.g. the usage-only chunk).
    - ``completion_tokens``: from ``usage`` when the server includes it.
    """

    is_data: bool
    done: bool
    content: str | None = None
    completion_tokens: int | None = None


def parse_sse_chunk(line: str) -> SSEChunk:
    """Parse one SSE line. Raises ValueError on a malformed ``data:`` JSON payload.

    Blank lines and SSE comments (``: ping`` keepalives) yield ``is_data=False``.
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith(_DATA_PREFIX):
        return SSEChunk(is_data=False, done=False)

    payload = stripped[len(_DATA_PREFIX):].strip()
    if payload == _DONE_PAYLOAD:
        return SSEChunk(is_data=True, done=True)

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed SSE data payload: {payload!r}") from e

    choices = obj.get("choices") or []
    content = choices[0].get("delta", {}).get("content") if choices else None
    usage = obj.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return SSEChunk(is_data=True, done=False, content=content, completion_tokens=tokens)


def _s2ms(x: float | None) -> float | None:
    return None if x is None else x * 1000.0


def assemble_row(
    start: float,
    chunk_times: Sequence[float],
    output_tokens: int | None,
    ok: bool,
) -> dict:
    """Per-request raw-CSV row from chunk arrival timestamps (all seconds).

    ``start`` is the request-send time; ``chunk_times`` are arrival times of each
    content-bearing chunk. Timing boundaries come from ``analyze.summarize_stream``
    (first chunk = TTFT, consecutive gaps = ITL); here they are just converted to
    milliseconds. ``ok`` marks request success (got >=1 content chunk, no error).
    """
    stats = summarize_stream(start, chunk_times, output_tokens=output_tokens)
    return {
        "start_ts": start,
        "ttft_ms": _s2ms(stats.ttft_s),
        "itl_mean_ms": _s2ms(stats.itl_mean_s),
        "total_ms": _s2ms(stats.total_s),
        "n_output_tokens": stats.n_output_tokens,
        "ok": ok,
    }


def select_context(messages: Sequence[dict]) -> list[dict] | None:
    """Context up to (excluding) the final assistant turn; None if unusable.

    Mirrors the M9 / M8-demo prompt construction: drop a trailing assistant turn so
    the context ends with a user message. Returns ``None`` when the result would
    not end with a user turn (so the locustfile can skip it).
    """
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    if not msgs or msgs[-1].get("role") != "user":
        return None
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def _context_length(messages: Sequence[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def order_by_length_buckets(
    items: Sequence[tuple[str, list[dict]]], n_buckets: int = 3
) -> list[tuple[str, list[dict]]]:
    """Reorder ``(id, messages)`` so consecutive picks rotate short/mid/long.

    Sort by context length, split into ``n_buckets`` contiguous length buckets, then
    round-robin across buckets. Deterministic (stable sort), so a tier rotates
    context lengths instead of hammering one size. Fewer items than buckets just
    returns the length-sorted list.
    """
    items = list(items)
    if len(items) <= 1 or n_buckets <= 1:
        return items
    ordered = sorted(items, key=lambda it: _context_length(it[1]))
    k = min(n_buckets, len(ordered))
    # Contiguous near-equal buckets (numpy.array_split style).
    n = len(ordered)
    base, extra = divmod(n, k)
    buckets: list[list[tuple[str, list[dict]]]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        buckets.append(ordered[start:start + size])
        start += size
    out: list[tuple[str, list[dict]]] = []
    for j in range(max(len(b) for b in buckets)):
        for b in buckets:
            if j < len(b):
                out.append(b[j])
    return out
