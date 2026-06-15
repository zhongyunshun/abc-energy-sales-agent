"""Unit tests for M5 behaviour-probe construction + diff rendering (T5.3).

Pure logic, no GPU. Covers fixture loading/validation, prompt construction (reuses
the unit-tested ChatML template), the empty-think strip, and the before/after
markdown assembly (including the length-mismatch guard).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sales_agent.common.schema import Message
from sales_agent.training.dpo_probes import (
    Probe,
    build_probe_prompt,
    count_changed,
    load_probes,
    render_behavior_diff,
    strip_think_prefix,
)
from sales_agent.training.formatting import RESPONSE_PART

FIXTURES = Path(__file__).parent / "fixtures"
PROBES = str(FIXTURES / "dpo_probes.jsonl")


# --- load_probes ------------------------------------------------------------


def test_fixture_has_20_balanced_probes():
    probes = load_probes(PROBES)
    assert len(probes) == 20
    cats = [p.category for p in probes]
    assert cats.count("pushy") == 10
    assert cats.count("rate_hallucination") == 10
    # every context ends with a user turn (load_probes enforces it)
    assert all(p.context[-1].role == "user" for p in probes)
    assert all(p.last_user for p in probes)


def test_load_rejects_non_user_ending(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "p", "category": "pushy", "context": ['
        '{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="end with a user message"):
        load_probes(str(bad))


def test_load_rejects_empty_context(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "p", "category": "pushy", "context": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_probes(str(bad))


# --- build_probe_prompt -----------------------------------------------------


def test_prompt_ends_with_generation_prompt():
    probe = Probe("p1", "pushy", [Message(role="user", content="Let me think it over.")])
    prompt = build_probe_prompt(probe)
    assert prompt.endswith(RESPONSE_PART)
    assert "Let me think it over." in prompt


def test_prompt_default_system_injected_when_absent():
    probe = Probe("p1", "pushy", [Message(role="user", content="Hi")])
    prompt = build_probe_prompt(probe, default_system="SYS")
    assert prompt.startswith("<|im_start|>system\nSYS<|im_end|>\n")


def test_prompt_default_system_not_duplicated():
    probe = Probe(
        "p1", "pushy",
        [Message(role="system", content="Existing"), Message(role="user", content="Hi")],
    )
    prompt = build_probe_prompt(probe, default_system="SYS")
    assert prompt.count("<|im_start|>system\n") == 1
    assert "Existing" in prompt and "SYS" not in prompt


def test_all_fixture_probes_build_prompts():
    for probe in load_probes(PROBES):
        prompt = build_probe_prompt(probe)
        assert prompt.endswith(RESPONSE_PART)
        assert probe.last_user in prompt


# --- strip_think_prefix -----------------------------------------------------


def test_strip_think_prefix_removes_empty_block():
    assert strip_think_prefix("<think>\n\n</think>\n\nHello there.") == "Hello there."
    assert strip_think_prefix("  <think></think> Quoted reply.") == "Quoted reply."


def test_strip_think_prefix_leaves_normal_text():
    assert strip_think_prefix("Of course, happy to help.") == "Of course, happy to help."


# --- render_behavior_diff ---------------------------------------------------


def test_render_behavior_diff_contains_both_sides():
    probes = [
        Probe("p1", "pushy", [Message(role="user", content="Call me back.")]),
        Probe("p2", "rate_hallucination", [Message(role="user", content="What's my rate?")]),
    ]
    before = ["You'd regret missing this, sign now!", "It's 9 cents per kWh, guaranteed."]
    after = ["Of course, when suits you?", "It depends on usage; I won't guess a number."]
    md = render_behavior_diff(probes, before, after, meta={"beta": 0.1, "lr": "5e-6"})
    assert "# M5 DPO behaviour diff — 2 probes" in md
    assert "beta: 0.1" in md and "lr: 5e-6" in md
    assert "pushy × 1" in md and "rate_hallucination × 1" in md
    for probe in probes:
        assert probe.id in md
        assert probe.last_user in md
    for text in before + after:
        assert text in md


def test_render_behavior_diff_strips_think_in_outputs():
    probes = [Probe("p1", "pushy", [Message(role="user", content="hi")])]
    md = render_behavior_diff(probes, ["<think></think>\nPolite."], ["<think></think>\nGrounded."])
    assert "Polite." in md and "Grounded." in md
    assert "<think>" not in md


def test_render_behavior_diff_length_mismatch_raises():
    probes = [Probe("p1", "pushy", [Message(role="user", content="hi")])]
    with pytest.raises(ValueError, match="length mismatch"):
        render_behavior_diff(probes, ["a", "b"], ["c"])


def test_count_changed_ignores_think_prefix():
    before = ["<think></think>\nSame.", "Pushy.", "Grounded."]
    after = ["Same.", "Less pushy.", "Grounded."]
    # probe 0 identical after strip, probe 1 changed, probe 2 identical -> 1
    assert count_changed(before, after) == 1


def test_render_behavior_diff_flags_identical_and_changed():
    probes = [
        Probe("p1", "pushy", [Message(role="user", content="a")]),
        Probe("p2", "pushy", [Message(role="user", content="b")]),
    ]
    md = render_behavior_diff(probes, ["same", "old"], ["same", "new"])
    assert "changed on **1/2** probes" in md
    assert "## p1 (pushy) — _identical_" in md
    assert "## p2 (pushy)\n" in md  # changed -> no identical flag


def test_render_behavior_diff_appends_conclusion():
    probes = [Probe("p1", "pushy", [Message(role="user", content="a")])]
    md = render_behavior_diff(
        probes, ["x"], ["y"], conclusion="Margins converged; behaviour mixed."
    )
    assert "## Conclusion" in md
    assert "Margins converged; behaviour mixed." in md


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
