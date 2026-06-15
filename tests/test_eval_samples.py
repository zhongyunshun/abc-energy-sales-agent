"""Unit tests for M9 eval-sample construction + reasoning stripping (T9.1).

Boundaries pinned: multi-turn / single-turn / with-system prompt construction,
malformed records (no trailing assistant, empty), the leading <think> strip
(empty / non-empty / absent / mid-reply), and deterministic stratified sampling.
"""

from __future__ import annotations

import pytest

from sales_agent.evals.samples import (
    EvalSample,
    build_eval_samples,
    select_samples,
    strip_reasoning,
)


def _rec(messages, *, rid="dlg-x", scenario="general"):
    return {"id": rid, "scenario": scenario, "lang": "en", "messages": messages}


# --- build_eval_samples: prompt / gold boundaries --------------------------


def test_single_turn():
    rec = _rec(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ]
    )
    (s,) = build_eval_samples([rec])
    assert s.prompt_messages == [{"role": "user", "content": "Hi"}]
    assert s.gold == "Hello, how can I help?"
    assert s.prompt_messages[-1]["role"] == "user"


def test_multi_turn_keeps_earlier_assistant_turns():
    rec = _rec(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2 (gold)"},
        ]
    )
    (s,) = build_eval_samples([rec])
    # Prompt is everything before the LAST assistant turn (earlier a1 retained).
    assert [m["role"] for m in s.prompt_messages] == ["user", "assistant", "user"]
    assert s.prompt_messages[1]["content"] == "a1"
    assert s.gold == "a2 (gold)"


def test_with_system_message_preserved_in_prompt():
    rec = _rec(
        [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ]
    )
    (s,) = build_eval_samples([rec])
    assert s.prompt_messages[0] == {"role": "system", "content": "You are an agent."}
    assert s.prompt_messages[-1]["role"] == "user"
    assert s.gold == "Hello."


def test_scenario_and_meta_carried():
    rec = _rec(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}],
        rid="dlg-42",
        scenario="info_gathering",
    )
    rec["meta"] = {"synth_model": "m"}
    (s,) = build_eval_samples([rec])
    assert isinstance(s, EvalSample)
    assert s.id == "dlg-42"
    assert s.scenario == "info_gathering"
    assert s.meta == {"synth_model": "m"}


def test_default_scenario_general_when_missing():
    rec = {"id": "d", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]}
    (s,) = build_eval_samples([rec])
    assert s.scenario == "general"


# --- build_eval_samples: malformed records raise (no silent drop) -----------


def test_last_not_assistant_raises():
    rec = _rec([{"role": "user", "content": "Hi"}])
    with pytest.raises(ValueError, match="expected assistant"):
        build_eval_samples([rec])


def test_empty_messages_raises():
    with pytest.raises(ValueError, match="messages missing or empty"):
        build_eval_samples([_rec([])])


def test_empty_gold_raises():
    rec = _rec(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "   "}]
    )
    with pytest.raises(ValueError, match="final assistant content is empty"):
        build_eval_samples([rec])


def test_prompt_not_ending_in_user_raises():
    # system, assistant -> after dropping the assistant the prompt ends with system.
    rec = _rec(
        [{"role": "system", "content": "S"}, {"role": "assistant", "content": "A"}]
    )
    with pytest.raises(ValueError, match="prompt must end with a user turn"):
        build_eval_samples([rec])


# --- strip_reasoning -------------------------------------------------------


def test_strip_empty_think():
    assert strip_reasoning("<think>\n\n</think>\n\nHello there.") == "Hello there."


def test_strip_whitespace_then_think():
    assert strip_reasoning("   <think>  </think>  Answer") == "Answer"


def test_strip_nonempty_think():
    assert strip_reasoning("<think>let me reason</think>The answer is X.") == "The answer is X."


def test_strip_no_think_unchanged():
    assert strip_reasoning("Just a normal reply.") == "Just a normal reply."


def test_strip_only_leading_think_not_midreply():
    # A think tag that is not at the very start must be preserved.
    text = "Real answer. <think>aside</think>"
    assert strip_reasoning(text) == "Real answer. <think>aside</think>"


def test_strip_think_only_becomes_empty():
    assert strip_reasoning("<think></think>") == ""


# --- select_samples: deterministic stratified sampling ---------------------


def _samples(counts: dict[str, int]) -> list[EvalSample]:
    out = []
    for sc, k in counts.items():
        for i in range(k):
            out.append(EvalSample(id=f"{sc}-{i}", scenario=sc, prompt_messages=[
                {"role": "user", "content": "u"}], gold="g"))
    return out


def test_select_none_returns_all():
    samples = _samples({"a": 3, "b": 2})
    assert select_samples(samples, None, seed=42) == samples


def test_select_n_ge_pool_returns_all():
    samples = _samples({"a": 3})
    assert len(select_samples(samples, 10, seed=42)) == 3


def test_select_zero_returns_empty():
    assert select_samples(_samples({"a": 3}), 0, seed=42) == []


def test_select_quota_sums_to_n_and_stratified():
    samples = _samples({"a": 60, "b": 30, "c": 10})  # 100 total
    picked = select_samples(samples, 50, seed=42)
    assert len(picked) == 50
    by = {}
    for s in picked:
        by[s.scenario] = by.get(s.scenario, 0) + 1
    # Proportional: 50% of each scenario.
    assert by == {"a": 30, "b": 15, "c": 5}


def test_select_deterministic_same_seed():
    samples = _samples({"a": 60, "b": 30, "c": 10})
    ids1 = [s.id for s in select_samples(samples, 50, seed=7)]
    ids2 = [s.id for s in select_samples(samples, 50, seed=7)]
    assert ids1 == ids2


def test_select_independent_of_model_under_test():
    # The batch depends only on (samples, n, seed): two callers (the three groups)
    # get identical ids -- the DoD "same batch" guarantee.
    samples = _samples({"a": 40, "b": 60})
    assert [s.id for s in select_samples(samples, 33, seed=1)] == [
        s.id for s in select_samples(samples, 33, seed=1)
    ]
