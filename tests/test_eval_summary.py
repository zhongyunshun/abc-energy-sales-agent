"""Unit tests for M9 summary aggregation (T9.3 summary): per-scenario trigger
rates + length-token distribution, with numeric assertions and empty handling."""

from __future__ import annotations

from sales_agent.evals.rules import RULE_NAMES
from sales_agent.evals.summary import summarize_results


def _row(scenario, *, n_tokens, **flag_overrides):
    flags = {name: False for name in RULE_NAMES}
    flags.update(flag_overrides)
    return {"id": f"r{n_tokens}", "scenario": scenario, "rule_flags": flags, "n_tokens": n_tokens}


def test_overall_rule_rates():
    rows = [
        _row("info_gathering", n_tokens=10, made_up_price=True),
        _row("info_gathering", n_tokens=20, made_up_price=True),
        _row("objection_handling", n_tokens=30),
        _row("objection_handling", n_tokens=40),
    ]
    s = summarize_results(rows, model_tag="dpo", gen_config={"temperature": 0.0})
    assert s["model_tag"] == "dpo"
    assert s["n_samples"] == 4
    # 2 of 4 flagged made_up_price -> 0.5.
    assert s["overall"]["rule_rates"]["made_up_price"] == 0.5
    assert s["overall"]["rule_counts"]["made_up_price"] == 2
    assert s["overall"]["rule_rates"]["over_length"] == 0.0


def test_by_scenario_grouping_and_rates():
    rows = [
        _row("info_gathering", n_tokens=10, no_question_in_gathering=True),
        _row("info_gathering", n_tokens=20),
        _row("objection_handling", n_tokens=30, role_break=True),
    ]
    s = summarize_results(rows, model_tag="sft", gen_config={})
    ig = s["by_scenario"]["info_gathering"]
    assert ig["n"] == 2
    assert ig["rule_rates"]["no_question_in_gathering"] == 0.5
    oh = s["by_scenario"]["objection_handling"]
    assert oh["n"] == 1
    assert oh["rule_rates"]["role_break"] == 1.0
    # no_question rule never fires outside gathering -> 0 here.
    assert oh["rule_rates"]["no_question_in_gathering"] == 0.0


def test_length_percentiles():
    rows = [_row("general", n_tokens=t) for t in [10, 20, 30, 40, 50]]
    s = summarize_results(rows, model_tag="base", gen_config={})
    lt = s["overall"]["length_tokens"]
    assert lt["min"] == 10
    assert lt["max"] == 50
    assert lt["mean"] == 30.0
    assert lt["p50"] == 30.0  # median of 10..50


def test_scenarios_sorted():
    rows = [_row("objection_handling", n_tokens=5), _row("closing", n_tokens=5),
            _row("info_gathering", n_tokens=5)]
    s = summarize_results(rows, model_tag="x", gen_config={})
    assert list(s["by_scenario"]) == ["closing", "info_gathering", "objection_handling"]


def test_empty_results():
    s = summarize_results([], model_tag="x", gen_config={})
    assert s["n_samples"] == 0
    assert s["overall"]["n"] == 0
    assert s["overall"]["rule_rates"] == {}
    assert s["overall"]["length_tokens"] == {}
    assert s["by_scenario"] == {}


def test_gen_config_passthrough():
    s = summarize_results([_row("general", n_tokens=5)], model_tag="x",
                          gen_config={"temperature": 0.0, "max_tokens": 256})
    assert s["gen_config"] == {"temperature": 0.0, "max_tokens": 256}
