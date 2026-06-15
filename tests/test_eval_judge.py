"""Unit tests for M10 LLM-as-a-Judge pure logic + mocked orchestration (T10.2/3/4).

Pinned: same-id selection across the three groups (the DoD guarantee), robust
parsing of every bad-response mode (non-JSON / missing field / out-of-range /
bad type), model_tag x scenario aggregation with the "no significant difference"
marking, and the async run_judge over the conftest fake (zero real API calls).
"""

from __future__ import annotations

import json

import jinja2
import pytest

from sales_agent.common.openrouter import OpenRouterClient
from sales_agent.evals.judge import (
    JudgeConfig,
    JudgeParseError,
    JudgeScore,
    aggregate_scores,
    build_judge_prompt,
    estimate_cost,
    format_dialogue,
    parse_judge_response,
    run_judge,
    select_judge_samples,
)

GOOD = json.dumps(
    {
        "coherence": {"score": 4, "reason": "addresses the question"},
        "sales_logic": {"score": 3, "reason": "ok but generic"},
        "professionalism": {"score": 5, "reason": "compliant"},
        "hallucination": {"score": 5, "reason": "no invented price"},
    }
)


def _row(rid: str, scenario: str, completion: str = "c") -> dict:
    return {
        "id": rid,
        "scenario": scenario,
        "prompt_messages": [
            {"role": "system", "content": "You are an ABC Energy agent."},
            {"role": "user", "content": f"question for {rid}"},
        ],
        "completion": completion,
    }


def _cfg(**over) -> JudgeConfig:
    base = {
        "seed": 42,
        "judge_models": ["judgeA"],
        "sampling": {"n_samples": 100},
        "temperature": 0.0,
        "max_tokens": 64,
        "max_retries": 2,
        "client_max_retries": 0,
        "concurrency": 4,
        "smoke": {"n_samples": 5},
        "no_diff_threshold": 0.3,
        "pricing": {},
    }
    base.update(over)
    return JudgeConfig.from_dict(base)


def _tiny_template() -> jinja2.Template:
    return jinja2.Environment(autoescape=False).from_string(
        "{{ dialogue }}||{{ candidate }}||{{ scenario }}"
    )


# --- T10.2 select_judge_samples: SAME ids across groups ---------------------


def _three_groups(n_ids: int = 10):
    """Same ids, same scenarios, different per-tag completions; dpo order shuffled."""
    scen = ["general"] * 6 + ["objection_handling"] * 4
    base = [_row(f"d{i}", scen[i], f"base-{i}") for i in range(n_ids)]
    sft = [_row(f"d{i}", scen[i], f"sft-{i}") for i in range(n_ids)]
    dpo = [_row(f"d{i}", scen[i], f"dpo-{i}") for i in range(n_ids)]
    dpo = list(reversed(dpo))  # different on-disk order must not matter
    return {"base": base, "sft": sft, "dpo": dpo}


def test_select_same_id_set_across_groups():
    sel = select_judge_samples(_three_groups(), n=5, seed=42)
    id_sets = [{r["id"] for r in rows} for rows in sel.values()]
    assert id_sets[0] == id_sets[1] == id_sets[2]
    assert all(len(rows) == 5 for rows in sel.values())


def test_select_preserves_each_groups_own_rows():
    sel = select_judge_samples(_three_groups(), n=5, seed=42)
    # Same id order across tags, but each tag keeps ITS OWN completion.
    assert [r["id"] for r in sel["base"]] == [r["id"] for r in sel["dpo"]]
    for r in sel["base"]:
        assert r["completion"].startswith("base-")
    for r in sel["dpo"]:
        assert r["completion"].startswith("dpo-")


def test_select_stratified_quota():
    sel = select_judge_samples(_three_groups(), n=5, seed=42)
    counts = {}
    for r in sel["base"]:
        counts[r["scenario"]] = counts.get(r["scenario"], 0) + 1
    assert counts == {"general": 3, "objection_handling": 2}  # 5 * 6/10, 5 * 4/10


def test_select_deterministic_same_seed():
    a = select_judge_samples(_three_groups(), n=5, seed=7)
    b = select_judge_samples(_three_groups(), n=5, seed=7)
    assert [r["id"] for r in a["base"]] == [r["id"] for r in b["base"]]


def test_select_n_ge_pool_returns_all_common():
    sel = select_judge_samples(_three_groups(8), n=100, seed=42)
    assert all(len(rows) == 8 for rows in sel.values())


def test_select_zero_returns_empty():
    sel = select_judge_samples(_three_groups(), n=0, seed=42)
    assert all(rows == [] for rows in sel.values())


def test_select_uses_only_common_ids():
    g = _three_groups(10)
    g["dpo"] = g["dpo"][:7]  # dpo missing 3 ids -> only the common 7 are eligible
    sel = select_judge_samples(g, n=100, seed=42)
    common = {r["id"] for r in g["dpo"]}
    assert all({r["id"] for r in rows} == common for rows in sel.values())


def test_select_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        select_judge_samples({}, n=5, seed=42)


def test_select_empty_tag_raises():
    with pytest.raises(ValueError, match="no result rows"):
        select_judge_samples({"base": [_row("d0", "general")], "sft": []}, n=1, seed=42)


def test_select_row_without_id_raises():
    with pytest.raises(ValueError, match="without an 'id'"):
        select_judge_samples({"base": [{"scenario": "general"}]}, n=1, seed=42)


def test_select_no_common_ids_raises():
    g = {"base": [_row("a", "general")], "sft": [_row("b", "general")]}
    with pytest.raises(ValueError, match="no common ids"):
        select_judge_samples(g, n=1, seed=42)


# --- T10.3a parse_judge_response: every bad-response mode -------------------


def test_parse_legal():
    out = parse_judge_response(GOOD)
    assert out["scores"] == {
        "coherence": 4, "sales_logic": 3, "professionalism": 5, "hallucination": 5,
    }
    assert out["rationale"]["coherence"] == "addresses the question"


def test_parse_bare_int_dims():
    raw = json.dumps(
        {"coherence": 4, "sales_logic": 3, "professionalism": 5, "hallucination": 2}
    )
    out = parse_judge_response(raw)
    assert out["scores"]["hallucination"] == 2
    assert out["rationale"]["coherence"] == ""  # no reason supplied


def test_parse_markdown_fenced():
    raw = f"```json\n{GOOD}\n```"
    assert parse_judge_response(raw)["scores"]["coherence"] == 4


def test_parse_missing_dimension():
    raw = json.dumps(
        {"coherence": {"score": 4}, "sales_logic": {"score": 3}, "professionalism": {"score": 5}}
    )
    err = parse_judge_response(raw)
    assert isinstance(err, JudgeParseError) and err.kind == "missing_field"
    assert "hallucination" in err.detail


def test_parse_out_of_range_high():
    raw = json.dumps(
        {"coherence": 6, "sales_logic": 3, "professionalism": 5, "hallucination": 5}
    )
    err = parse_judge_response(raw)
    assert isinstance(err, JudgeParseError) and err.kind == "out_of_range"


def test_parse_out_of_range_zero():
    raw = json.dumps(
        {"coherence": 0, "sales_logic": 3, "professionalism": 5, "hallucination": 5}
    )
    err = parse_judge_response(raw)
    assert isinstance(err, JudgeParseError) and err.kind == "out_of_range"


def test_parse_bad_type_string():
    raw = json.dumps(
        {"coherence": "high", "sales_logic": 3, "professionalism": 5, "hallucination": 5}
    )
    err = parse_judge_response(raw)
    assert isinstance(err, JudgeParseError) and err.kind == "bad_type"


def test_parse_bad_type_fractional():
    raw = json.dumps(
        {"coherence": 3.5, "sales_logic": 3, "professionalism": 5, "hallucination": 5}
    )
    err = parse_judge_response(raw)
    assert isinstance(err, JudgeParseError) and err.kind == "bad_type"


def test_parse_bool_rejected():
    raw = json.dumps(
        {"coherence": True, "sales_logic": 3, "professionalism": 5, "hallucination": 5}
    )
    err = parse_judge_response(raw)
    assert isinstance(err, JudgeParseError) and err.kind == "bad_type"


def test_parse_whole_float_accepted():
    raw = json.dumps(
        {"coherence": 4.0, "sales_logic": 3, "professionalism": 5, "hallucination": 5}
    )
    assert parse_judge_response(raw)["scores"]["coherence"] == 4


def test_parse_not_json():
    err = parse_judge_response("the reply is pretty good overall")
    assert isinstance(err, JudgeParseError) and err.kind == "not_json"


def test_parse_empty():
    err = parse_judge_response("")
    assert isinstance(err, JudgeParseError) and err.kind == "not_json"


# --- prompt construction ---------------------------------------------------


def test_format_dialogue_maps_roles():
    txt = format_dialogue(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    )
    assert "[System] S" in txt and "[Customer] U" in txt and "[Agent] A" in txt


def test_build_judge_prompt_blind_and_shaped():
    tmpl = _tiny_template()
    msgs = build_judge_prompt(_row("d0", "closing", completion="Sure, let's proceed."), tmpl)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Sure, let's proceed." in msgs[1]["content"]
    # Blind: build_judge_prompt never receives a model_tag, so it can't leak one.
    assert "base" not in msgs[1]["content"] and "dpo" not in msgs[1]["content"]


def test_real_template_renders():

    from sales_agent.common.config import find_repo_root
    from sales_agent.evals.judge import load_template

    tmpl = load_template(find_repo_root() / "configs" / "prompts" / "judge.j2")
    msgs = build_judge_prompt(_row("d0", "objection_handling", "We offer flexible terms."), tmpl)
    body = msgs[1]["content"]
    assert "We offer flexible terms." in body
    assert "coherence" in body and "hallucination" in body
    assert "JSON" in body


# --- T10.3b aggregate_scores ----------------------------------------------


def _score(rid, tag, sc, vals, jm="judgeA"):
    dims = ["coherence", "sales_logic", "professionalism", "hallucination"]
    return JudgeScore(
        id=rid, model_tag=tag, scenario=sc,
        scores=dict(zip(dims, vals, strict=True)),
        rationale={d: "" for d in dims},
        judge_model=jm, judge_raw="{}",
    )


def test_aggregate_overall_means_and_order():
    scores = [
        _score("a", "base", "general", [2, 2, 2, 2]),
        _score("b", "base", "general", [2, 2, 2, 2]),
        _score("a", "sft", "general", [5, 5, 5, 5]),
        _score("b", "sft", "general", [5, 5, 5, 5]),
        _score("a", "dpo", "general", [5, 4, 5, 5]),
        _score("b", "dpo", "general", [5, 4, 5, 5]),
    ]
    t = aggregate_scores(scores)
    assert t["model_tags"] == ["base", "sft", "dpo"]  # canonical order
    j = t["judges"]["judgeA"]
    assert j["overall"]["base"]["coherence"]["mean"] == 2.0
    assert j["overall"]["sft"]["coherence"]["mean"] == 5.0
    assert j["overall"]["dpo"]["sales_logic"]["mean"] == 4.0
    assert j["overall"]["base"]["coherence"]["std"] == 0.0
    assert j["overall"]["base"]["coherence"]["n"] == 2


def test_aggregate_no_diff_marking():
    scores = [
        _score("a", "base", "general", [2, 2, 2, 2]),
        _score("a", "sft", "general", [5, 5, 5, 5]),
        _score("a", "dpo", "general", [5, 5, 5, 5]),  # dpo == sft on coherence
    ]
    j = aggregate_scores(scores, no_diff_threshold=0.3)["judges"]["judgeA"]
    pw = {(p["a"], p["b"]): p for p in j["pairwise"]["coherence"]}
    assert pw[("sft", "dpo")]["no_diff"] is True  # diff 0 < 0.3
    assert pw[("base", "sft")]["no_diff"] is False  # diff 3 >= 0.3


def test_aggregate_by_scenario_split():
    scores = [
        _score("a", "sft", "general", [5, 5, 5, 5]),
        _score("b", "sft", "closing", [4, 4, 4, 4]),
    ]
    j = aggregate_scores(scores)["judges"]["judgeA"]
    assert j["by_scenario"]["general"]["sft"]["coherence"]["mean"] == 5.0
    assert j["by_scenario"]["closing"]["sft"]["coherence"]["mean"] == 4.0


def test_aggregate_groups_by_judge():
    scores = [
        _score("a", "sft", "general", [5, 5, 5, 5], jm="judgeA"),
        _score("a", "sft", "general", [3, 3, 3, 3], jm="judgeB"),
    ]
    t = aggregate_scores(scores)
    assert set(t["judges"]) == {"judgeA", "judgeB"}
    assert t["judges"]["judgeA"]["overall"]["sft"]["coherence"]["mean"] == 5.0
    assert t["judges"]["judgeB"]["overall"]["sft"]["coherence"]["mean"] == 3.0


# --- estimate_cost ---------------------------------------------------------


def test_estimate_cost_priced_and_missing():
    tokens = {
        "m1": {"requests": 2, "prompt_tokens": 1000, "completion_tokens": 200},
        "m2": {"requests": 1, "prompt_tokens": 500, "completion_tokens": 100},
    }
    pricing = {"m1": {"input_per_m": 3.0, "output_per_m": 15.0}}
    cost = estimate_cost(tokens, pricing)
    assert cost["per_model"]["m1"]["usd"] == pytest.approx(0.006)  # 1000*3/1e6 + 200*15/1e6
    assert cost["per_model"]["m2"]["usd"] is None
    assert cost["missing_pricing"] == ["m2"]
    assert cost["total_usd"] == pytest.approx(0.006)


# --- T10.4 run_judge over the conftest fake (zero real API calls) ----------


def _client(fake):
    return OpenRouterClient(model="judgeA", raw_client=fake, backoff_base=0, max_retries=0)


async def test_run_judge_scores_all_blind(fake_openrouter):
    fake_openrouter.set_default(GOOD)
    cfg = _cfg(judge_models=["judgeA", "judgeB"])
    samples = {"base": [_row("d0", "general", "b")], "sft": [_row("d0", "general", "s")]}
    run = await run_judge(samples, _client(fake_openrouter), cfg, _tiny_template())
    # 2 judges x 2 tags x 1 sample = 4 scores.
    assert run.succeeded == 4 and run.attempted == 4
    assert {s.judge_model for s in run.scores} == {"judgeA", "judgeB"}
    assert {s.model_tag for s in run.scores} == {"base", "sft"}
    # Per-model token totals accumulated for the cost estimate.
    assert run.tokens_by_model["judgeA"]["requests"] == 2
    assert run.tokens_by_model["judgeB"]["prompt_tokens"] > 0


async def test_run_judge_parse_failure_counted_after_retries(fake_openrouter):
    # max_retries=2 -> 3 attempts; all non-JSON -> one counted parse failure, no crash.
    fake_openrouter.queue_responses("garbage", "still garbage", "nope")
    cfg = _cfg(max_retries=2)
    samples = {"base": [_row("d0", "general")]}
    run = await run_judge(samples, _client(fake_openrouter), cfg, _tiny_template())
    assert run.succeeded == 0
    assert run.parse_failures == 1
    assert run.failures_by_kind == {"not_json": 1}
    assert run.validation_retries == 2


async def test_run_judge_recovers_within_retries(fake_openrouter):
    fake_openrouter.queue_responses("garbage", GOOD)  # bad then good within retry budget
    cfg = _cfg(max_retries=2)
    samples = {"base": [_row("d0", "general")]}
    run = await run_judge(samples, _client(fake_openrouter), cfg, _tiny_template())
    assert run.succeeded == 1 and run.parse_failures == 0
    assert run.validation_retries == 1
