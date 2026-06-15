"""Unit tests for M5 preference-pair -> DPO format conversion (T5.1).

Pure logic, no GPU/tokenizer. Covers: the standard-format shape (prompt ends with
the assistant generation prompt; chosen/rejected are the raw replies), optional
default-system injection, the shared contract fixtures, and -- the boundary the
design doc (section 3-M5) explicitly requires -- a ValueError when the context
does not end with a user message (and when it is empty).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sales_agent.common.schema import Message, PreferencePair
from sales_agent.training.formatting import (
    IM_END,
    RESPONSE_PART,
    preference_pair_to_dpo,
    render_chatml,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _pair(
    context: list[Message], chosen: str = "good", rejected: str = "bad", **kw
) -> PreferencePair:
    base = dict(id="pref-test", scenario="pushy", context=context, chosen=chosen, rejected=rejected)
    base.update(kw)
    return PreferencePair(**base)


# --- happy path -------------------------------------------------------------


def test_standard_format_shape():
    pair = _pair(
        [Message(role="user", content="What's the rate?")],
        chosen="It depends.",
        rejected="9.5c/kWh.",
    )
    row = preference_pair_to_dpo(pair)
    assert set(row) == {"prompt", "chosen", "rejected"}
    # Completions are the raw reply strings, verbatim (no markers).
    assert row["chosen"] == "It depends."
    assert row["rejected"] == "9.5c/kWh."
    # Prompt ends with the assistant generation prompt: the model continues here.
    assert row["prompt"].endswith(RESPONSE_PART)
    assert IM_END not in row["prompt"].split(RESPONSE_PART)[-1]  # nothing after the header
    assert "What's the rate?" in row["prompt"]


def test_prompt_plus_chosen_is_coherent_chatml():
    pair = _pair([Message(role="user", content="Q")], chosen="A grounded answer.")
    row = preference_pair_to_dpo(pair)
    # prompt + chosen reconstructs the assistant turn body exactly.
    assert (row["prompt"] + row["chosen"]).endswith(
        "<|im_start|>assistant\nA grounded answer."
    )


def test_no_system_prompt_starts_with_user():
    pair = _pair([Message(role="user", content="Hi")])
    row = preference_pair_to_dpo(pair)
    assert row["prompt"].startswith("<|im_start|>user\nHi<|im_end|>\n")


def test_existing_system_preserved():
    pair = _pair(
        [Message(role="system", content="Be honest."), Message(role="user", content="Rate?")]
    )
    row = preference_pair_to_dpo(pair)
    assert row["prompt"].startswith("<|im_start|>system\nBe honest.<|im_end|>\n")


def test_default_system_injected_when_absent():
    pair = _pair([Message(role="user", content="Hi")])
    row = preference_pair_to_dpo(pair, default_system="SYS")
    assert row["prompt"].startswith("<|im_start|>system\nSYS<|im_end|>\n")
    # equivalent to rendering the augmented message list directly
    expected = render_chatml(
        [Message(role="system", content="SYS"), Message(role="user", content="Hi")],
        add_generation_prompt=True,
    )
    assert row["prompt"] == expected


def test_default_system_not_duplicated():
    pair = _pair(
        [Message(role="system", content="Existing"), Message(role="user", content="Hi")]
    )
    row = preference_pair_to_dpo(pair, default_system="SYS")
    assert row["prompt"].count("<|im_start|>system\n") == 1
    assert "Existing" in row["prompt"] and "SYS" not in row["prompt"]


def test_multi_turn_context_renders_all_turns():
    pair = _pair(
        [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello, how can I help?"),
            Message(role="user", content="Is now a good time? No, I'm busy."),
        ]
    )
    row = preference_pair_to_dpo(pair)
    assert "Hello, how can I help?" in row["prompt"]
    assert row["prompt"].endswith(RESPONSE_PART)


# --- shared contract fixtures -----------------------------------------------


def test_valid_fixtures_all_convert():
    lines = (FIXTURES / "preference_pairs_valid.jsonl").read_text(encoding="utf-8").splitlines()
    pairs = [PreferencePair.model_validate(json.loads(line)) for line in lines if line.strip()]
    assert len(pairs) >= 5
    for pair in pairs:
        row = preference_pair_to_dpo(pair)
        assert row["prompt"].endswith(RESPONSE_PART)
        assert row["chosen"] == pair.chosen and row["rejected"] == pair.rejected
        # last user message text must be present in the rendered prompt
        last_user = next(m.content for m in reversed(pair.context) if m.role == "user")
        assert last_user in row["prompt"]


# --- the required error boundary --------------------------------------------
#
# PreferencePair's schema rejects a non-user-ending context at construction, so
# we use model_construct to bypass validation and exercise the conversion's OWN
# defensive guard (design doc section 3-M5).


def test_context_not_ending_with_user_raises():
    bad = PreferencePair.model_construct(
        id="pref-bad",
        scenario="rate_hallucination",
        context=[
            Message(role="user", content="What's the rate?"),
            Message(role="assistant", content="It depends on usage."),
        ],
        chosen="A grounded reply.",
        rejected="A made-up rate.",
        meta={},
    )
    with pytest.raises(ValueError, match="end with a user message"):
        preference_pair_to_dpo(bad)


def test_empty_context_raises():
    bad = PreferencePair.model_construct(
        id="pref-empty", scenario="pushy", context=[], chosen="c", rejected="r", meta={}
    )
    with pytest.raises(ValueError, match="must not be empty"):
        preference_pair_to_dpo(bad)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
