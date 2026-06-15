"""M11 bench orchestrator (design doc 3-M11, task T11.2 + drives T11.3).

Runs the concurrency ladder against the live vLLM endpoint, one tier at a time, by
launching the locustfile headless in a SUBPROCESS per tier. Using a subprocess
(not in-process Locust) keeps gevent's monkey-patching contained -- it never
collides with this orchestrator -- and isolates each tier's raw CSV. After the
ladder it aggregates the raw CSVs into bench_summary.csv (design-doc 2.4), renders
the three PNGs, and writes manifest.json + a short bench_report.md.

This is a thin shell: all timing/aggregation/plotting logic is unit-tested in
sales_agent.bench.{locust_logic,report}. This script only orchestrates and is
validated against the real server (smoke then full ladder).

Exit codes (design doc 1.4):
  0  success
  2  input-contract failure (test set missing / no prompts)
  3  external-dependency failure (endpoint unreachable / a locust tier failed to run)

The 32 tier deliberately exceeds M8 max_num_seqs=16: vLLM queues past 16, so TTFT
rises and throughput plateaus -- that knee is the real curve. If a tier OOMs /
crashes the server (locust subprocess errors out, not mere request failures), this
exits 3 with the tier noted; do NOT raise the server cap to hide it.

Usage (after `bash scripts/serving/serve.sh`):
  uv run python scripts/bench/run_bench.py --config configs/bench.yaml            # full ladder
  uv run python scripts/bench/run_bench.py --config configs/bench.yaml --smoke    # 30s single tier
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from sales_agent.bench.report import (
    PLOT_FILES,
    SUMMARY_CSV_NAME,
    collect_summaries,
    raw_csv_path,
    summarize_from_raw,
    write_plots,
    write_summary_csv,
)
from sales_agent.common.config import find_repo_root, load_config
from sales_agent.common.manifest import build_manifest, write_manifest

logger = logging.getLogger("bench.run_bench")

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_DEPENDENCY = 3

LOCUSTFILE = "scripts/bench/locustfile.py"
REPORT_MD = "bench_report.md"

# Documented gap (M11 special note 3): FP16 cannot be served on the 4070 (12GB).
FP16_GAP_NOTE = (
    "FP16 vs INT4 TTFT/ITL comparison NOT measured: the merged FP16 model (8.045GB) "
    "does not fit alongside the display on the 4070 (~7.9GB free), so only the INT4 "
    "(AWQ/compressed-tensors) service is benchmarked here. Size/theory comparison: "
    "M7 manifest (8.045GB FP16 -> 2.666GB INT4, 3.02x); quality: M7's 5 FP16-vs-INT4 "
    "probes (no visible regression). Handed to M12 README."
)


def _endpoint_ready(host: str, health_route: str, timeout_s: float = 5.0) -> bool:
    url = host.rstrip("/") + health_route
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 local URL
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _resolve_plan(cfg: dict, smoke: bool) -> tuple[list[int], int, int]:
    """Return (tiers, run_time_s, warmup_s), applying the smoke override if set."""
    lad = cfg["ladder"]
    if smoke:
        s = cfg["smoke"]
        return list(s["concurrency"]), int(s["run_time_s"]), int(s["warmup_s"])
    return list(lad["concurrency"]), int(lad["run_time_s"]), int(lad["warmup_s"])


def _run_tier(
    concurrency: int, run_time_s: int, host: str, config_path: Path, raw_path: Path, repo_root: Path
) -> int:
    """Launch one headless Locust tier in a subprocess. Returns its exit code."""
    env = dict(os.environ)
    env["BENCH_CONFIG"] = str(config_path)
    env["BENCH_RAW_CSV"] = str(raw_path)
    cmd = [
        sys.executable, "-m", "locust",
        "-f", LOCUSTFILE,
        "--headless",
        "-u", str(concurrency),
        "-r", str(concurrency),     # spawn all users within ~1s
        "-t", f"{run_time_s}s",
        "--host", host,
        # We capture request failures ourselves (error_rate); failed requests must
        # NOT make Locust exit non-zero, or an expected-degradation tier reads as a crash.
        "--exit-code-on-error", "0",
    ]
    logger.info("tier c=%d: %s", concurrency, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, env=env)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bench.yaml")
    parser.add_argument("--host", default=None, help="override endpoint base URL (no /v1)")
    parser.add_argument("--smoke", action="store_true", help="one short tier (smoke block)")
    parser.add_argument("--output-dir", default=None, help="override reports output dir")
    args = parser.parse_args(argv)

    repo_root = find_repo_root()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    host = args.host or cfg["endpoint"]["host"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    tiers, run_time_s, warmup_s = _resolve_plan(cfg, args.smoke)

    # --- pre-flight: test set (contract) + endpoint reachability (dependency) ---
    test_path = Path(cfg["test_path"])
    if not test_path.exists():
        logger.error("test set not found at %s (M3 product). Cannot build prompts.", test_path)
        return EXIT_CONTRACT
    if not _endpoint_ready(host, cfg["endpoint"]["health_route"]):
        logger.error("endpoint not ready at %s%s. Run `bash scripts/serving/serve.sh` first.",
                     host, cfg["endpoint"]["health_route"])
        return EXIT_DEPENDENCY

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ladder=%s run_time=%ds warmup=%ds host=%s -> %s%s",
                tiers, run_time_s, warmup_s, host, output_dir, " [SMOKE]" if args.smoke else "")

    # --- run the ladder, tier by tier ---
    wall_start = time.time()
    for c in tiers:
        raw_path = raw_csv_path(output_dir, c)
        rc = _run_tier(c, run_time_s, host, config_path, raw_path, repo_root)
        if rc != 0:
            logger.error("tier c=%d locust subprocess failed (rc=%d) -- possible OOM/crash. "
                         "STOP and report (M11 note 2); not raising the server cap.", c, rc)
            return EXIT_DEPENDENCY
        if not raw_path.exists():
            logger.error("tier c=%d produced no raw CSV at %s", c, raw_path)
            return EXIT_DEPENDENCY
        s = summarize_from_raw(raw_path, c, run_time_s, warmup_s)
        logger.info(
            "tier c=%d: n=%d ok=%d fail=%d | ttft p50/p95=%s/%s ms | itl p50=%s ms | "
            "%.1f tok/s | %.2f req/s | err=%.1f%%",
            c, s.n_total, s.n_ok, s.n_failed,
            _f(s.ttft_p50), _f(s.ttft_p95), _f(s.itl_p50),
            s.throughput_tok_s, s.req_s, s.error_rate * 100,
        )
    wall_s = time.time() - wall_start

    # --- aggregate -> CSV + plots + manifest + report ---
    summaries = collect_summaries(output_dir, tiers, run_time_s, warmup_s)
    summary_csv = write_summary_csv(summaries, output_dir / SUMMARY_CSV_NAME)
    plots = write_plots(summaries, output_dir)
    (output_dir / REPORT_MD).write_text(_render_report(summaries, host, smoke=args.smoke),
                                        encoding="utf-8", newline="\n")

    manifest = build_manifest(
        inputs=[test_path, config_path],
        config={"tiers": tiers, "run_time_s": run_time_s, "warmup_s": warmup_s,
                "host": host, "smoke": args.smoke, "generation": cfg["generation"]},
        stats={
            "wall_seconds": round(wall_s, 1),
            "tiers": [s.as_dict() for s in summaries],
            "fp16_comparison_gap": FP16_GAP_NOTE,
        },
        repo_root=repo_root,
    )
    manifest_path = write_manifest(output_dir, manifest)

    logger.info("done in %.0fs: %s , %s , %s , %s , %s , manifest %s",
                wall_s, summary_csv, plots["throughput"], plots["ttft"], plots["itl"],
                output_dir / REPORT_MD, manifest_path)
    return EXIT_OK


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def _render_report(summaries, host: str, smoke: bool) -> str:
    title = "M11 load test (Locust, AWQ/INT4)" + (" -- SMOKE" if smoke else "")
    lines = [
        f"# {title}\n",
        f"- endpoint: `{host}` | served INT4 (compressed-tensors) on RTX 4070, M8 service",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
        "## Steady-state metrics vs concurrency\n",
        "| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | "
        "tok/s | req/s | error % | n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda t: t.concurrency):
        lines.append(
            f"| {s.concurrency} | {_f(s.ttft_p50)} | {_f(s.ttft_p95)} | {_f(s.itl_p50)} | "
            f"{_f(s.itl_p95)} | {s.throughput_tok_s:.1f} | {s.req_s:.2f} | "
            f"{s.error_rate * 100:.1f} | {s.n_total} |"
        )
    lines += [
        "\n## Plots\n",
        f"- ![throughput]({PLOT_FILES['throughput']})",
        f"- ![ttft]({PLOT_FILES['ttft']})",
        f"- ![itl]({PLOT_FILES['itl']})\n",
        "## Notes\n",
        "- Concurrency is closed-loop (N users, no think-time) so in-flight load ~= N. "
        "32 exceeds the M8 cap max_num_seqs=16 on purpose: vLLM queues past 16, so the "
        "16->32 step is where TTFT rises and throughput plateaus (the real knee).",
        f"- {FP16_GAP_NOTE}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
