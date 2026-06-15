"""Unit tests for M4 chat-template rendering and completion-only masking.

Pure logic, no GPU/tokenizer. Covers the correctness core (the masking contract,
open item 1): with/without system, single- and multi-turn, and exact assistant
masking boundaries -- only assistant turns contribute to loss; system/user are
masked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sales_agent.common.schema import DialogueRecord, Message
from sales_agent.training.formatting import (
    IM_END,
    INSTRUCTION_PART,
    RESPONSE_PART,
    assistant_content_spans,
    completion_only_spans,
    render_chatml,
    to_conversation,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _msgs(*pairs: tuple[str, str]) -> list[Message]:
    return [Message(role=r, content=c) for r, c in pairs]


def _record(messages: list[Message], **kw) -> DialogueRecord:
    n_turns = sum(1 for m in messages if m.role == "assistant")
    base = dict(
        id="dlg-test", source="synthetic:v1", scenario="general", lang="en", n_turns=n_turns
    )
    base.update(kw)
    return DialogueRecord(messages=messages, **base)


# --- render_chatml ----------------------------------------------------------


def test_render_no_system_single_turn():
    text = render_chatml(_msgs(("user", "Hi there"), ("assistant", "Hello!")))
    assert text == (
        "<|im_start|>user\nHi there<|im_end|>\n"
        "<|im_start|>assistant\nHello!<|im_end|>\n"
    )


def test_render_with_system_multi_turn():
    text = render_chatml(
        _msgs(
            ("system", "You are an energy sales agent."),
            ("user", "Your rates seem high."),
            ("assistant", "May I ask what you pay now?"),
            ("user", "About 80 a month."),
            ("assistant", "Thanks, I can prepare a comparison."),
        )
    )
    assert text.startswith("<|im_start|>system\nYou are an energy sales agent.<|im_end|>\n")
    assert INSTRUCTION_PART in text
    assert RESPONSE_PART in text
    assert text.endswith("<|im_start|>assistant\nThanks, I can prepare a comparison.<|im_end|>\n")


def test_render_add_generation_prompt():
    text = render_chatml(_msgs(("user", "Hi")), add_generation_prompt=True)
    assert text.endswith(RESPONSE_PART)
    # No assistant content yet after the generation prompt.
    assert text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"


# --- to_conversation --------------------------------------------------------


def test_to_conversation_strips_to_role_content():
    rec = _record(_msgs(("user", "Hi"), ("assistant", "Hello")), id="dlg-x", meta={"k": "v"})
    conv = to_conversation(rec)
    assert conv == {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }


def test_to_conversation_default_system_injected_when_absent():
    rec = _record(_msgs(("user", "Hi"), ("assistant", "Hello")))
    conv = to_conversation(rec, default_system="SYS")
    assert conv["messages"][0] == {"role": "system", "content": "SYS"}
    assert len(conv["messages"]) == 3


def test_to_conversation_default_system_not_duplicated():
    rec = _record(_msgs(("system", "Existing"), ("user", "Hi"), ("assistant", "Hello")))
    conv = to_conversation(rec, default_system="SYS")
    # Record already has a system message -> injection skipped, no duplication.
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert conv["messages"][0]["content"] == "Existing"


# --- completion-only masking boundaries -------------------------------------


def test_mask_single_turn_no_system():
    text = render_chatml(_msgs(("user", "Question?"), ("assistant", "Answer.")))
    spans = completion_only_spans(text)
    assert len(spans) == 1
    covered = text[spans[0][0]:spans[0][1]]
    # Loss span = assistant content + its closing end-of-turn marker.
    assert covered == "Answer.<|im_end|>\n"
    # The user turn must NOT be covered.
    assert "Question?" not in covered


def test_mask_multi_turn_only_assistant_in_loss():
    text = render_chatml(
        _msgs(
            ("system", "SYS"),
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
        )
    )
    spans = completion_only_spans(text)
    assert len(spans) == 2  # one per assistant turn
    covered = [text[s:e] for s, e in spans]
    assert covered == ["A1<|im_end|>\n", "A2<|im_end|>\n"]

    # Nothing outside the spans may contain assistant content; everything that
    # IS masked must include system + both user turns.
    masked = _complement(text, spans)
    assert "SYS" in masked and "U1" in masked and "U2" in masked
    # The assistant header marker itself is masked (only content is in loss).
    assert RESPONSE_PART in masked
    assert "A1" not in masked and "A2" not in masked


def test_mask_content_spans_exact_boundaries():
    text = render_chatml(
        _msgs(("user", "U1"), ("assistant", "first"), ("user", "U2"), ("assistant", "second"))
    )
    spans = assistant_content_spans(text)
    assert [text[s:e] for s, e in spans] == ["first", "second"]
    # Content spans exclude the end-of-turn marker (unlike loss spans).
    for s, e in spans:
        assert IM_END not in text[s:e]


def test_mask_with_system_unaffected_count():
    # System present vs absent must not change the number/content of loss spans.
    with_sys = render_chatml(_msgs(("system", "S"), ("user", "U"), ("assistant", "A")))
    no_sys = render_chatml(_msgs(("user", "U"), ("assistant", "A")))
    assert len(completion_only_spans(with_sys)) == len(completion_only_spans(no_sys)) == 1
    assert with_sys[slice(*completion_only_spans(with_sys)[0])] == "A<|im_end|>\n"


def test_mask_no_assistant_yields_no_spans():
    # An inference prompt (generation prompt appended, no answer) has no loss.
    text = render_chatml(_msgs(("user", "U")), add_generation_prompt=True)
    assert completion_only_spans(text) == []


def test_fixture_records_render_and_mask():
    # Reuse the shared contract fixtures: one with system (synthetic), one
    # without (hf real, single-turn). Both must render and mask cleanly.
    records = [
        DialogueRecord.model_validate(json.loads(line))
        for line in (FIXTURES / "dialogues_valid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen_system, seen_no_system = False, False
    for rec in records:
        text = render_chatml(rec.messages)
        spans = completion_only_spans(text)
        assert len(spans) == rec.n_turns  # one loss span per assistant turn
        if rec.messages[0].role == "system":
            seen_system = True
        else:
            seen_no_system = True
    assert seen_system and seen_no_system  # fixtures cover both branches


def _complement(text: str, spans: list[tuple[int, int]]) -> str:
    """Return the concatenation of all characters NOT inside any span."""
    out, prev = [], 0
    for s, e in spans:
        out.append(text[prev:s])
        prev = e
    out.append(text[prev:])
    return "".join(out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
