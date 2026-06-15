"""M8 concurrency demo (the M8 contract, task T8.4): proof that vLLM's continuous
batching serves many streamed requests far faster than running them back-to-back.

What it does:
  1. load N prompts from the test set (each = a dialogue's context up to, but not
     including, the final assistant turn) -- real sales contexts, not synthetic;
  2. fire one warm-up/baseline request serially to measure single-request latency;
  3. fire N streaming requests concurrently with asyncio, timing every chunk;
  4. compute per-request TTFT / total latency (sales_agent.bench.analyze), the
     batching speedup (N * single_latency vs concurrent wall-clock), and aggregate
     output throughput;
  5. print a summary table and commit the evidence under reports/serving/.

This is a thin shell over the unit-tested timing core; it is real-server / GPU
validation (run after serve.sh), not a unit test. Exit codes (the CLI contract):
0 success, 2 input-contract failure (no prompts), 3 external dep (endpoint
unreachable). The model emits an empty <think></think> that the serve-layer qwen3
reasoning parser strips into reasoning_content, so message.content is clean here.

Usage (after `bash scripts/serving/serve.sh`):
  uv run python scripts/serving/concurrency_demo.py --config configs/serve.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from sales_agent.bench.analyze import (
    aggregate_streams,
    batching_speedup,
    summarize_stream,
    throughput_tok_s,
)
from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl

logger = logging.getLogger("concurrency_demo")

EXIT_OK = 0
EXIT_CONTRACT = 2  # no usable prompts
EXIT_DEPENDENCY = 3  # endpoint unreachable

DEFAULT_TEST_PATH = "data/processed/test.jsonl"


@dataclass
class RequestResult:
    id: str
    stats: object  # StreamStats
    text: str


def build_prompts(test_path: Path, n: int, seed: int) -> list[tuple[str, list[dict]]]:
    """Take N dialogues' context (messages up to the last assistant turn).

    Returns (id, messages) where messages ends with a user turn -- the same prompt
    construction M9 uses (the M9 contract). Deterministic: first N records by file
    order (the split is already shuffled with a fixed seed upstream); ``seed`` is
    accepted for signature stability / future sampling.
    """
    prompts: list[tuple[str, list[dict]]] = []
    for rec in read_jsonl(test_path):
        msgs = rec["messages"]
        # Drop the trailing assistant turn -> context ends with the user message.
        if msgs and msgs[-1]["role"] == "assistant":
            ctx = msgs[:-1]
        else:
            ctx = msgs
        if not ctx or ctx[-1]["role"] != "user":
            continue
        prompts.append((rec["id"], [{"role": m["role"], "content": m["content"]} for m in ctx]))
        if len(prompts) >= n:
            break
    return prompts


async def stream_one(client, model: str, messages: list[dict], max_tokens: int) -> RequestResult:
    """Issue one streaming chat completion; time each content-bearing chunk."""
    start = time.perf_counter()
    chunk_times: list[float] = []
    parts: list[str] = []
    completion_tokens: int | None = None

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.0,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        now = time.perf_counter()
        if chunk.choices:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:  # time to first/next *visible* output token
                chunk_times.append(now)
                parts.append(piece)
        usage = getattr(chunk, "usage", None)
        if usage is not None and getattr(usage, "completion_tokens", None) is not None:
            completion_tokens = usage.completion_tokens

    # Fall back to chunk count when the server omits usage (vLLM streams ~1 tok/chunk).
    out_tokens = completion_tokens if completion_tokens is not None else len(chunk_times)
    stats = summarize_stream(start, chunk_times, output_tokens=out_tokens)
    return RequestResult(id="", stats=stats, text="".join(parts))


async def run_demo(endpoint: str, model: str, prompts, max_tokens: int) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY")

    # 1. Warm-up (discarded): the FIRST generation request after startup pays a
    #    one-time JIT cost (vLLM inductor/CUDA-graph capture for the decode shapes),
    #    which would inflate a cold baseline and exaggerate the speedup. Run one
    #    throwaway request so the measured baseline reflects steady-state latency.
    base_id, base_msgs = prompts[0]
    logger.info("warm-up: 1 throwaway request (absorb first-request JIT cost) ...")
    await stream_one(client, model, base_msgs, max_tokens)

    # 2. Baseline: one warm serial request -> representative single-request latency.
    logger.info("baseline: 1 warm serial request to measure single-request latency ...")
    baseline = await stream_one(client, model, base_msgs, max_tokens)
    single_latency = baseline.stats.total_s or 0.0
    logger.info("baseline single-request latency: %.3fs", single_latency)

    # 3. Concurrent: all N requests at once; wall-clock around the gather.
    logger.info("concurrent: firing %d streaming requests ...", len(prompts))
    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *(stream_one(client, model, msgs, max_tokens) for _, msgs in prompts)
    )
    wall = time.perf_counter() - wall_start
    for (pid, _), r in zip(prompts, results, strict=True):
        r.id = pid

    stats_list = [r.stats for r in results]
    agg = aggregate_streams(stats_list)
    speedup = batching_speedup(single_latency, wall, len(prompts)) if single_latency else None
    tput = throughput_tok_s(agg.total_output_tokens, wall) if agg.total_output_tokens else 0.0

    # Throughput-based batching evidence (robust to per-request token-count variance):
    # aggregate tok/s under N-way load vs a single warm stream's tok/s.
    base_tokens = baseline.stats.n_output_tokens or 0
    single_stream_tok_s = (
        throughput_tok_s(base_tokens, single_latency) if single_latency and base_tokens else None
    )
    throughput_gain = (
        round(tput / single_stream_tok_s, 2) if single_stream_tok_s else None
    )

    return {
        "endpoint": endpoint,
        "model": model,
        "n_requests": len(prompts),
        "max_tokens": max_tokens,
        "single_latency_s": round(single_latency, 4),
        "single_stream_tok_s": round(single_stream_tok_s, 2) if single_stream_tok_s else None,
        "concurrent_wall_s": round(wall, 4),
        "serial_estimate_s": round(single_latency * len(prompts), 4),
        "batching_speedup": round(speedup, 3) if speedup else None,
        "throughput_tok_s": round(tput, 2),
        "throughput_gain_vs_single": throughput_gain,
        "aggregate": agg.as_dict(),
        "per_request": [
            {
                "id": r.id,
                "ttft_s": round(r.stats.ttft_s, 4) if r.stats.ttft_s is not None else None,
                "total_s": round(r.stats.total_s, 4) if r.stats.total_s is not None else None,
                "itl_mean_s": round(r.stats.itl_mean_s, 5)
                if r.stats.itl_mean_s is not None
                else None,
                "n_chunks": r.stats.n_chunks,
                "n_output_tokens": r.stats.n_output_tokens,
            }
            for r in results
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def render_report(summary: dict) -> str:
    """Human-readable Markdown (committed for M12 reference)."""
    lines = [
        "# M8 concurrency demo (continuous batching)\n",
        f"- endpoint: `{summary['endpoint']}` | model: `{summary['model']}`",
        f"- requests: {summary['n_requests']} concurrent (streaming), "
        f"max_tokens={summary['max_tokens']}",
        f"- generated_at: {summary['generated_at']}\n",
        "## Batching efficiency\n",
        f"- warm single-request latency (baseline): **{summary['single_latency_s']}s** "
        f"(~{summary['single_stream_tok_s']} tok/s single stream)",
        f"- serial estimate ({summary['n_requests']}x single): "
        f"**{summary['serial_estimate_s']}s**",
        f"- concurrent wall-clock ({summary['n_requests']} streams): "
        f"**{summary['concurrent_wall_s']}s**",
        f"- **latency speedup: {summary['batching_speedup']}x**  "
        f"({summary['n_requests']}x single / concurrent wall)",
        f"- aggregate output throughput: **{summary['throughput_tok_s']} tok/s**",
        f"- **throughput gain vs single stream: {summary['throughput_gain_vs_single']}x**  "
        f"(robust continuous-batching evidence)\n",
        "## Latency summary (per-request rollup)\n",
        "| metric | mean | p50 | p95 |",
        "|---|---:|---:|---:|",
    ]
    a = summary["aggregate"]

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    lines.append(f"| TTFT (s) | {fmt(a['ttft_mean_s'])} | {fmt(a['ttft_p50_s'])} "
                 f"| {fmt(a['ttft_p95_s'])} |")
    lines.append(f"| total (s) | {fmt(a['total_mean_s'])} | {fmt(a['total_p50_s'])} "
                 f"| {fmt(a['total_p95_s'])} |")
    lines.append("\n## Per-request\n")
    lines.append("| id | TTFT (s) | total (s) | ITL mean (s) | tokens |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in summary["per_request"]:
        lines.append(
            f"| {r['id']} | {fmt(r['ttft_s'])} | {fmt(r['total_s'])} "
            f"| {fmt(r['itl_mean_s'])} | {r['n_output_tokens']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serve.yaml")
    parser.add_argument("--endpoint", default=None, help="override OpenAI base URL (…/v1)")
    parser.add_argument("--test-path", default=None, help="override test.jsonl path")
    parser.add_argument("--n", type=int, default=16, help="concurrent request count")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="reports/serving")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg["seed"]
    server = cfg["server"]
    model = cfg["model"]["served_name"]
    endpoint = args.endpoint or f"http://127.0.0.1:{server['port']}/v1"

    test_path = Path(args.test_path or DEFAULT_TEST_PATH)
    if not test_path.exists():
        logger.error("test set not found at %s (M3 product). Cannot build prompts.", test_path)
        return EXIT_CONTRACT
    prompts = build_prompts(test_path, args.n, seed)
    if not prompts:
        logger.error("no usable prompts (context ending in user) in %s", test_path)
        return EXIT_CONTRACT
    if len(prompts) < args.n:
        logger.warning("only %d prompts available (< requested %d)", len(prompts), args.n)

    try:
        summary = asyncio.run(run_demo(endpoint, model, prompts, args.max_tokens))
    except Exception as e:  # noqa: BLE001 -- surface endpoint failure as exit 3
        logger.error("demo failed against %s: %s", endpoint, e)
        logger.error("Is the server up? Run `bash scripts/serving/serve.sh` first.")
        return EXIT_DEPENDENCY

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "concurrency_demo.json"
    md_path = out_dir / "concurrency_demo.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_report(summary), encoding="utf-8", newline="\n")

    logger.info(
        "done: %d concurrent in %.2fs vs %.2fs serial estimate -> %sx speedup | "
        "%.1f tok/s | evidence -> %s , %s",
        summary["n_requests"], summary["concurrent_wall_s"], summary["serial_estimate_s"],
        summary["batching_speedup"], summary["throughput_tok_s"], json_path, md_path,
    )
    return EXIT_OK


if __name__ == "__main__":
    import sys

    sys.exit(main())
