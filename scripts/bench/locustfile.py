"""M11 Locust user: stream the vLLM chat endpoint and log per-request timing.

Thin shell over the unit-tested pure logic in ``sales_agent.bench.locust_logic``.
It is NOT unit-tested (it drives a real server); the locustfile's *computation*
(SSE parsing, first-chunk-=-TTFT / gap-=-ITL timing, prompt-pool construction) is
all in ``locust_logic`` and tested there.

Each virtual user fires streaming ``/v1/chat/completions`` requests back-to-back
(closed loop, wait_time 0) so in-flight load stays ~= the user count. Per request
it records the chunk arrival times, writes ONE row to the per-tier raw CSV
(start_ts, ttft_ms, itl_mean_ms, total_ms, n_output_tokens, ok), and fires custom
Locust events (ttft_ms / itl_ms) for live console visibility. The orchestrator
(run_bench.py) drives this headless once per concurrency tier and reads the raw
CSVs back via ``sales_agent.bench.report``.

Driven by env vars (set by run_bench.py; sensible defaults for a manual run):
  BENCH_CONFIG   path to configs/bench.yaml          (default: configs/bench.yaml)
  BENCH_RAW_CSV  where to write this tier's raw rows  (default: reports/bench/raw_adhoc.csv)

Manual single-tier run (after serve.sh):
  BENCH_RAW_CSV=reports/bench/raw_8.csv \
    uv run locust -f scripts/bench/locustfile.py --headless -u 8 -r 8 -t 60s \
      --host http://127.0.0.1:8000
"""

from __future__ import annotations

import csv
import itertools
import logging
import os
import threading
import time
from pathlib import Path

from locust import HttpUser, constant, events, task

from sales_agent.bench.locust_logic import (
    RAW_CSV_COLUMNS,
    assemble_row,
    order_by_length_buckets,
    parse_sse_chunk,
    select_context,
)
from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl

logger = logging.getLogger("bench.locustfile")

# --- shared state, initialised once per Locust process (events.init) --------
_cfg: dict = {}
_prompts: list[tuple[str, list[dict]]] = []
_prompt_cycle = None
_csv_lock = threading.Lock()
_csv_file = None
_csv_writer = None


def _build_prompt_pool(test_path: Path) -> list[tuple[str, list[dict]]]:
    """Test dialogues' contexts (ending in user), length-bucket round-robined."""
    pool: list[tuple[str, list[dict]]] = []
    for rec in read_jsonl(test_path):
        ctx = select_context(rec.get("messages", []))
        if ctx is not None:
            pool.append((rec["id"], ctx))
    return order_by_length_buckets(pool, n_buckets=3)


@events.init.add_listener
def _on_init(environment, **_kw):
    global _cfg, _prompts, _prompt_cycle, _csv_file, _csv_writer
    _cfg = load_config(os.environ.get("BENCH_CONFIG", "configs/bench.yaml"))
    _prompts = _build_prompt_pool(Path(_cfg["test_path"]))
    if not _prompts:
        raise RuntimeError(f"no usable prompts in {_cfg['test_path']} (need user-ending context)")
    _prompt_cycle = itertools.cycle(_prompts)

    raw_path = Path(os.environ.get("BENCH_RAW_CSV", "reports/bench/raw_adhoc.csv"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _csv_file = open(raw_path, "w", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=RAW_CSV_COLUMNS)
    _csv_writer.writeheader()
    _csv_file.flush()
    logger.info("bench init: %d prompts, raw CSV -> %s", len(_prompts), raw_path)


@events.quitting.add_listener
def _on_quitting(environment, **_kw):
    global _csv_file
    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _csv_file = None


def _write_row(row: dict) -> None:
    with _csv_lock:
        _csv_writer.writerow(row)
        _csv_file.flush()


class BenchUser(HttpUser):
    # Closed loop: no think-time, so each user keeps exactly one request in flight.
    wait_time = constant(0)

    @task
    def stream_chat(self):
        prompt_id, messages = next(_prompt_cycle)
        gen = _cfg["generation"]
        payload = {
            "model": _cfg["endpoint"]["served_model"],
            "messages": messages,
            "stream": True,
            "temperature": gen["temperature"],
            "max_tokens": gen["max_tokens"],
            "stream_options": {"include_usage": True},
        }
        route = _cfg["endpoint"]["chat_route"]

        start = time.time()
        chunk_times: list[float] = []
        completion_tokens: int | None = None
        ok = False

        with self.client.post(
            route, json=payload, stream=True, catch_response=True, name="chat"
        ) as resp:
            try:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                else:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        chunk = parse_sse_chunk(line)
                        if chunk.done:
                            break
                        if chunk.completion_tokens is not None:
                            completion_tokens = chunk.completion_tokens
                        if chunk.content:  # a visible output token arrived
                            chunk_times.append(time.time())
                    ok = len(chunk_times) > 0
                    if ok:
                        resp.success()
                    else:
                        resp.failure("no content chunks")
            except Exception as e:  # noqa: BLE001 -- surface as a Locust failure + ok=False
                resp.failure(repr(e))

        out_tokens = completion_tokens if completion_tokens is not None else len(chunk_times)
        row = assemble_row(start, chunk_times, output_tokens=out_tokens, ok=ok)
        _write_row(row)

        # Custom events: surface TTFT / ITL in the live console (Locust's own "chat"
        # stat already tracks total latency / failures). Raw CSV stays the source of truth.
        if row["ttft_ms"] is not None:
            events.request.fire(
                request_type="METRIC", name="ttft_ms", response_time=row["ttft_ms"],
                response_length=out_tokens or 0, exception=None, context={},
            )
        if row["itl_mean_ms"] is not None:
            events.request.fire(
                request_type="METRIC", name="itl_ms", response_time=row["itl_mean_ms"],
                response_length=0, exception=None, context={},
            )
