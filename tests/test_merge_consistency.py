"""Unit tests for the M6 merge consistency comparator + prompt loader (T6.1/T6.3).

Pure logic, no GPU: exercises both match modes (exact / prefix_tokens), mismatch
detection and diff rendering, the ordering/length guards, and the fixed-prompt
loader's rendering + contract validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sales_agent.training.merge_consistency import (
    MODE_EXACT,
    MODE_PREFIX_TOKENS,
    ConsistencyResult,
    Generation,
    compare_generations,
    compare_one,
    load_consistency_prompts,
    render_consistency_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "merge_consistency_prompts.jsonl"


def _gen(pid: str, text: str, ids: list[int]) -> Generation:
    return Generation(prompt_id=pid, text=text, token_ids=ids)


# --- exact mode -------------------------------------------------------------


def test_exact_all_match():
    peft = [_gen("p0", "hello there", [1, 2, 3]), _gen("p1", "ok", [9])]
    merged = [_gen("p0", "hello there", [1, 2, 3]), _gen("p1", "ok", [9])]
    res = compare_generations(peft, merged, mode=MODE_EXACT)
    assert res.consistent is True
    assert res.n_mismatch == 0
    assert res.mismatched_ids == []
    assert res.summary()["mode"] == "exact"
    assert res.summary()["prefix_n"] is None


def test_exact_mismatch_detected_with_diff():
    peft = [_gen("p0", "hello there", [1, 2, 3])]
    merged = [_gen("p0", "hello world", [1, 2, 9])]
    res = compare_generations(peft, merged, mode=MODE_EXACT)
    assert res.consistent is False
    assert res.n_mismatch == 1
    assert res.mismatched_ids == ["p0"]
    diff = res.render_diffs()
    assert "MISMATCH" in diff and "p0" in diff
    assert "hello there" in diff and "hello world" in diff
    # char-level divergence index reported (texts differ at index 6: 'there'/'world')
    assert "first divergence at char 6" in diff


def test_match_verdict_has_empty_detail():
    v = compare_one(_gen("p0", "x", [1]), _gen("p0", "x", [1]), mode=MODE_EXACT)
    assert v.match is True
    assert v.detail == ""


# --- prefix_tokens mode -----------------------------------------------------


def test_prefix_tokens_relaxation_passes_late_drift():
    # Texts differ, but the first 64 token ids are identical (drift only at id 70).
    a = list(range(70)) + [100]
    b = list(range(70)) + [999]
    peft = [_gen("p0", "tail-A", a)]
    merged = [_gen("p0", "tail-B", b)]
    # exact mode would fail (text differs)...
    assert compare_generations(peft, merged, mode=MODE_EXACT).consistent is False
    # ...prefix_tokens/64 passes (divergence is past the window).
    res = compare_generations(peft, merged, mode=MODE_PREFIX_TOKENS, prefix_n=64)
    assert res.consistent is True
    assert res.summary()["prefix_n"] == 64


def test_prefix_tokens_early_divergence_fails():
    a = list(range(64))
    b = list(range(10)) + [777] + list(range(11, 64))
    peft = [_gen("p0", "A", a)]
    merged = [_gen("p0", "B", b)]
    res = compare_generations(peft, merged, mode=MODE_PREFIX_TOKENS, prefix_n=64)
    assert res.consistent is False
    diff = res.render_diffs()
    assert "first divergence at token 10" in diff
    assert "prefix_n=64" in diff


def test_prefix_tokens_shorter_than_n_still_compares_full():
    # Both shorter than prefix_n -> full sequence compared.
    peft = [_gen("p0", "A", [1, 2, 3])]
    merged = [_gen("p0", "B", [1, 2, 4])]
    res = compare_generations(peft, merged, mode=MODE_PREFIX_TOKENS, prefix_n=64)
    assert res.consistent is False


# --- guards -----------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="count mismatch"):
        compare_generations([_gen("p0", "x", [1])], [], mode=MODE_EXACT)


def test_out_of_order_prompt_ids_raise():
    peft = [_gen("p0", "x", [1])]
    merged = [_gen("p1", "x", [1])]
    with pytest.raises(ValueError, match="prompt id mismatch"):
        compare_generations(peft, merged, mode=MODE_EXACT)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown match mode"):
        compare_one(_gen("p0", "x", [1]), _gen("p0", "x", [1]), mode="fuzzy")


# --- ConsistencyResult summary shape ----------------------------------------


def test_summary_records_mismatched_ids():
    verdicts = compare_generations(
        [_gen("p0", "a", [1]), _gen("p1", "b", [2])],
        [_gen("p0", "a", [1]), _gen("p1", "B", [9])],
        mode=MODE_EXACT,
    )
    s = verdicts.summary()
    assert s["n_total"] == 2
    assert s["n_mismatch"] == 1
    assert s["consistent"] is False
    assert s["mismatched_ids"] == ["p1"]


# --- fixed-prompt loader (T6.3 logic) ---------------------------------------


def test_load_consistency_prompts_fixture_renders_8():
    prompts = load_consistency_prompts(str(PROMPTS_FIXTURE))
    assert len(prompts) == 8
    ids = [p.id for p in prompts]
    assert len(set(ids)) == 8  # unique ids
    for p in prompts:
        # render_chatml(add_generation_prompt=True) ends with the assistant header.
        assert p.rendered.endswith("<|im_start|>assistant\n")
        assert "<|im_start|>user\n" in p.rendered


def test_load_prompts_injects_default_system():
    no_sys = load_consistency_prompts(str(PROMPTS_FIXTURE))
    with_sys = load_consistency_prompts(str(PROMPTS_FIXTURE), default_system="You are helpful.")
    # default_system only injected where context had no system message.
    assert any("<|im_start|>system\nYou are helpful." in p.rendered for p in with_sys)
    assert len(no_sys) == len(with_sys)


def test_load_prompts_rejects_non_user_ending(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "context": [{"role": "user", "content": "hi"},
                                            {"role": "assistant", "content": "yo"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must end with a user message"):
        load_consistency_prompts(str(bad))


def test_load_prompts_rejects_empty_context(tmp_path):
    bad = tmp_path / "empty.jsonl"
    bad.write_text(json.dumps({"id": "x", "context": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_consistency_prompts(str(bad))


def test_consistency_result_is_frozen_dataclass():
    res = ConsistencyResult(mode=MODE_EXACT, prefix_n=64, verdicts=[])
    assert res.consistent is True  # vacuously consistent with no prompts
    assert res.n_total == 0


# --- committable evidence report --------------------------------------------


def test_render_report_pass_shows_output_and_ids():
    peft = [_gen("p0", "Sure, happy to help.", [1, 2]), _gen("p1", "Of course.", [3])]
    merged = [_gen("p0", "Sure, happy to help.", [1, 2]), _gen("p1", "Of course.", [3])]
    res = compare_generations(peft, merged, mode=MODE_EXACT)
    md = render_consistency_report(res, merged, meta={"base": "Qwen3-4B", "git_commit": "abc123"})
    assert "**PASS** — 2/2 prompts match" in md
    assert "p0 — PASS" in md and "p1 — PASS" in md
    assert "Sure, happy to help." in md  # the actual generation is shown as evidence
    assert "base: Qwen3-4B" in md and "git_commit: abc123" in md


def test_render_report_fail_shows_diff_block():
    peft = [_gen("p0", "professional reply", [1, 2, 3])]
    merged = [_gen("p0", "pushy reply", [1, 9, 3])]
    res = compare_generations(peft, merged, mode=MODE_EXACT)
    md = render_consistency_report(res, merged)
    assert "**FAIL** — 0/1 prompts match" in md
    assert "p0 — FAIL" in md
    assert "professional reply" in md and "pushy reply" in md  # diff detail embedded
