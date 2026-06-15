"""M11 re-aggregate + re-plot from existing raw CSVs (the M11 contract, task T11.3).

Thin shell over sales_agent.bench.report. Use it to regenerate bench_summary.csv
and the three PNGs from raw_{c}.csv files that run_bench.py already produced --
e.g. to tweak a plot without re-running the ~12-min GPU ladder. All logic is
unit-tested in sales_agent.bench.report.

Exit codes (the CLI contract): 0 success; 2 no raw CSVs found.

Usage:
  uv run python scripts/bench/plot_bench.py --config configs/bench.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sales_agent.bench.report import (
    SUMMARY_CSV_NAME,
    collect_summaries,
    write_plots,
    write_summary_csv,
)
from sales_agent.common.config import load_config

logger = logging.getLogger("bench.plot_bench")

EXIT_OK = 0
EXIT_CONTRACT = 2


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bench.yaml")
    parser.add_argument("--smoke", action="store_true", help="use the smoke tier/durations")
    parser.add_argument("--output-dir", default=None, help="override reports output dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    plan = cfg["smoke"] if args.smoke else cfg["ladder"]
    tiers = list(plan["concurrency"])
    run_time_s, warmup_s = int(plan["run_time_s"]), int(plan["warmup_s"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])

    summaries = collect_summaries(output_dir, tiers, run_time_s, warmup_s)
    if not summaries:
        logger.error("no raw_{c}.csv found under %s for tiers %s. Run run_bench.py first.",
                     output_dir, tiers)
        return EXIT_CONTRACT

    summary_csv = write_summary_csv(summaries, output_dir / SUMMARY_CSV_NAME)
    plots = write_plots(summaries, output_dir)
    logger.info("re-aggregated %d tiers -> %s , %s",
                len(summaries), summary_csv, ", ".join(str(p) for p in plots.values()))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
