"""Summary aggregation for offline eval (design doc 3-M9 step 5, contract 2.4).

Pure logic: roll a list of per-sample result rows into ``summary.json`` -- each
rule's trigger rate and the reply-length distribution, both overall and grouped by
scenario. Reuses :func:`sales_agent.bench.analyze.percentile` for the length
quantiles (the same percentile core M8/M11 use -- no second implementation).
"""

from __future__ import annotations

from collections.abc import Sequence

from sales_agent.bench.analyze import percentile
from sales_agent.evals.rules import RULE_NAMES


def _length_stats(lengths: Sequence[int]) -> dict:
    """Length distribution (token counts) for one group; {} when empty."""
    if not lengths:
        return {}
    return {
        "mean": round(sum(lengths) / len(lengths), 2),
        "p50": round(percentile(lengths, 50), 2),
        "p90": round(percentile(lengths, 90), 2),
        "p95": round(percentile(lengths, 95), 2),
        "min": min(lengths),
        "max": max(lengths),
    }


def _group_summary(rows: list[dict]) -> dict:
    """Trigger rates + length distribution for one group of result rows."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "rule_rates": {}, "rule_counts": {}, "length_tokens": {}}
    counts = {name: sum(1 for r in rows if r["rule_flags"].get(name)) for name in RULE_NAMES}
    rates = {name: round(counts[name] / n, 4) for name in RULE_NAMES}
    lengths = [r["n_tokens"] for r in rows]
    return {
        "n": n,
        "rule_rates": rates,
        "rule_counts": counts,
        "length_tokens": _length_stats(lengths),
    }


def summarize_results(results: list[dict], *, model_tag: str, gen_config: dict) -> dict:
    """Aggregate result rows into the M9 ``summary.json`` (design doc 2.4).

    Each row needs ``scenario``, ``rule_flags`` (the four booleans), and
    ``n_tokens``. Output: overall rollup + a per-scenario breakdown (sorted), each
    with rule trigger rates (and raw counts) and the length-token distribution.
    ``no_question_in_gathering`` is reported as a fraction of all rows in the group,
    so it reads as 0 in non-gathering scenarios -- interpretable and comparable.
    """
    by_scenario: dict[str, list[dict]] = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], []).append(r)
    return {
        "model_tag": model_tag,
        "n_samples": len(results),
        "gen_config": gen_config,
        "overall": _group_summary(results),
        "by_scenario": {sc: _group_summary(by_scenario[sc]) for sc in sorted(by_scenario)},
    }
