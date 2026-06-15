"""Unit tests for M1 format conversion (src/sales_agent/data/normalize.py)."""

from __future__ import annotations

import pytest

from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import DialogueRecord, Message, validate_dialogue
from sales_agent.data.normalize import (
    DROP_EMPTY,
    DROP_NON_ENGLISH,
    CleanConfig,
    SourceBatch,
    alpaca_to_dialogue,
    clean_dialogue,
    explode_prefixed_pairs,
    prefixed_pair_to_dialogue,
    redact_pii,
    run_pipeline,
    sharegpt_to_dialogue,
    tag_scenario,
)
from tests.conftest import FIXTURES_DIR

SOURCE = "test:source"
RULES = CleanConfig()


def make_record(*turns: tuple[str, str]) -> DialogueRecord:
    """Build a DialogueRecord from (role, content) pairs without validation."""
    msgs = [Message(role=r, content=c) for r, c in turns]
    return DialogueRecord(
        id="dlg-unset",
        source=SOURCE,
        scenario="general",
        lang="en",
        n_turns=sum(1 for m in msgs if m.role == "assistant"),
        messages=msgs,
    )


class TestAlpacaToDialogue:
    def test_valid_record_converts_to_single_turn_dialogue(self):
        raw = {
            "instruction": "Explain a fixed-rate plan.",
            "input": "",
            "output": "It locks in your unit price for the contract length.",
        }
        rec = alpaca_to_dialogue(raw, SOURCE)
        assert rec is not None
        assert rec.source == SOURCE
        assert rec.lang == "en"
        assert rec.n_turns == 1
        assert rec.scenario == "general"
        assert rec.meta["raw_format"] == "alpaca"
        assert [m.role for m in rec.messages] == ["user", "assistant"]
        assert rec.messages[0].content == "Explain a fixed-rate plan."
        assert validate_dialogue(rec) == []

    def test_input_field_is_appended_to_user_message(self):
        raw = {
            "instruction": "Respond to the objection.",
            "input": "Customer says: too expensive.",
            "output": "I understand price matters.",
        }
        rec = alpaca_to_dialogue(raw, SOURCE)
        assert rec is not None
        expected = "Respond to the objection.\n\nCustomer says: too expensive."
        assert rec.messages[0].content == expected

    def test_id_is_stable_content_hash(self):
        raw = {"instruction": "Hi there.", "output": "Hello, how can I help?"}
        a = alpaca_to_dialogue(raw, SOURCE)
        b = alpaca_to_dialogue(dict(raw), SOURCE)
        assert a is not None and b is not None
        assert a.id == b.id
        assert a.id.startswith("dlg-") and len(a.id) == len("dlg-") + 12

    @pytest.mark.parametrize(
        "raw",
        [
            {"instruction": "", "output": "No instruction."},
            {"input": "context only", "output": "Missing instruction key."},
            {"instruction": "Missing output key."},
            {"instruction": "Whitespace output.", "output": "   "},
            {"instruction": 123, "output": "Non-string instruction."},
            {"instruction": "Non-string output.", "output": ["not", "a", "string"]},
            {},
        ],
    )
    def test_invalid_records_return_none(self, raw):
        assert alpaca_to_dialogue(raw, SOURCE) is None

    def test_fixture_file_conversion_counts(self):
        rows = list(read_jsonl(FIXTURES_DIR / "raw_alpaca.jsonl"))
        converted = [alpaca_to_dialogue(r, SOURCE) for r in rows]
        ok = [r for r in converted if r is not None]
        failed = [row["note"] for row, rec in zip(rows, converted, strict=True) if rec is None]
        assert len(ok) == 5
        assert all(note.startswith("invalid") for note in failed)
        assert all(validate_dialogue(r) == [] for r in ok)


class TestSharegptToDialogue:
    def test_valid_record_with_system(self):
        raw = {
            "conversations": [
                {"from": "system", "value": "You are a sales agent."},
                {"from": "human", "value": "Hi there."},
                {"from": "gpt", "value": "Hello! How can I help with your energy plan?"},
            ]
        }
        rec = sharegpt_to_dialogue(raw, SOURCE)
        assert rec is not None
        assert rec.meta["raw_format"] == "sharegpt"
        assert [m.role for m in rec.messages] == ["system", "user", "assistant"]
        assert rec.n_turns == 1
        assert validate_dialogue(rec) == []

    def test_consecutive_same_role_messages_are_merged(self):
        raw = {
            "conversations": [
                {"from": "human", "value": "Hello?"},
                {"from": "gpt", "value": "Hi, this is Alex."},
                {"from": "gpt", "value": "Is now a good time?"},
            ]
        }
        rec = sharegpt_to_dialogue(raw, SOURCE)
        assert rec is not None
        assert [m.role for m in rec.messages] == ["user", "assistant"]
        assert rec.messages[1].content == "Hi, this is Alex.\nIs now a good time?"
        assert rec.n_turns == 1

    def test_mid_dialogue_system_is_hoisted_to_front(self):
        raw = {
            "conversations": [
                {"from": "human", "value": "Is switching complicated?"},
                {"from": "system", "value": "Keep replies short."},
                {"from": "gpt", "value": "Not at all."},
            ]
        }
        rec = sharegpt_to_dialogue(raw, SOURCE)
        assert rec is not None
        assert [m.role for m in rec.messages] == ["system", "user", "assistant"]
        assert rec.messages[0].content == "Keep replies short."
        assert validate_dialogue(rec) == []

    def test_multiple_system_messages_are_merged_at_front(self):
        raw = {
            "conversations": [
                {"from": "system", "value": "You are a sales agent."},
                {"from": "human", "value": "Hi."},
                {"from": "system", "value": "Be brief."},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        rec = sharegpt_to_dialogue(raw, SOURCE)
        assert rec is not None
        assert [m.role for m in rec.messages] == ["system", "user", "assistant"]
        assert rec.messages[0].content == "You are a sales agent.\nBe brief."

    def test_assistant_first_is_converted_but_fails_validation(self):
        # Conversion does not silently repair role order; validation catches it.
        raw = {
            "conversations": [
                {"from": "gpt", "value": "May I interest you in a plan?"},
                {"from": "human", "value": "No thanks."},
                {"from": "gpt", "value": "Have a great day!"},
            ]
        }
        rec = sharegpt_to_dialogue(raw, SOURCE)
        assert rec is not None
        assert validate_dialogue(rec) != []

    @pytest.mark.parametrize(
        "raw",
        [
            {"conversations": []},
            {"messages": [{"role": "user", "content": "wrong shape"}]},
            {"conversations": [{"from": "robot", "value": "Beep."}]},
            {"conversations": [{"from": "human", "value": 42}]},
            {"conversations": ["not a dict"]},
            {"conversations": [{"from": "system", "value": "Only system, no turns."}]},
            {},
        ],
    )
    def test_invalid_records_return_none(self, raw):
        assert sharegpt_to_dialogue(raw, SOURCE) is None

    def test_fixture_file_conversion_counts(self):
        rows = list(read_jsonl(FIXTURES_DIR / "raw_sharegpt.jsonl"))
        converted = [sharegpt_to_dialogue(r, SOURCE) for r in rows]
        failed = [row["note"] for row, rec in zip(rows, converted, strict=True) if rec is None]
        assert len([r for r in converted if r is not None]) == 5
        assert all(note.startswith("invalid") for note in failed)


class TestPrefixedPairs:
    def test_row_explodes_into_independent_pairs(self):
        row = {
            "0": "Customer: Hi, I need a new phone plan.",
            "1": "Salesman: Happy to help! What is your monthly budget?",
            "2": "Customer: Im looking for a laptop recommendation.",
            "3": "Salesman: Sure! What software will you run on it?",
            "4": None,
            "5": None,
        }
        pairs = explode_prefixed_pairs(row)
        assert pairs == [
            {
                "customer": "Hi, I need a new phone plan.",
                "salesman": "Happy to help! What is your monthly budget?",
            },
            {
                "customer": "Im looking for a laptop recommendation.",
                "salesman": "Sure! What software will you run on it?",
            },
        ]

    def test_column_order_is_numeric_not_lexicographic(self):
        # "10" must sort after "9", not between "1" and "2"
        row = {
            "9": "Customer: Question nine?",
            "10": "Salesman: Answer ten.",
            "0": "Customer: Question zero?",
            "1": "Salesman: Answer one.",
        }
        pairs = explode_prefixed_pairs(row)
        assert [p["customer"] for p in pairs] == ["Question zero?", "Question nine?"]

    def test_unpaired_and_unprefixed_cells_are_skipped(self):
        row = {
            "0": "Salesman: I start the row, no preceding customer.",
            "1": "Customer: Where is my answer?",
            "2": "Salesman: Right here.",
            "3": "no prefix at all",
            "4": "Customer: Trailing question without an answer.",
        }
        pairs = explode_prefixed_pairs(row)
        assert pairs == [{"customer": "Where is my answer?", "salesman": "Right here."}]

    def test_empty_row_yields_no_pairs(self):
        assert explode_prefixed_pairs({"0": "", "1": None, "note": "meta"}) == []

    def test_pair_converts_to_single_turn_dialogue(self):
        rec = prefixed_pair_to_dialogue(
            {"customer": "How do variable rates work?", "salesman": "They follow market prices."},
            SOURCE,
        )
        assert rec is not None
        assert [m.role for m in rec.messages] == ["user", "assistant"]
        assert rec.n_turns == 1
        assert rec.meta["raw_format"] == "prefixed_pairs"
        assert validate_dialogue(rec) == []

    @pytest.mark.parametrize(
        "raw",
        [
            {"customer": "", "salesman": "Answer."},
            {"customer": "Question?", "salesman": "   "},
            {"customer": "Question?"},
            {"salesman": "Answer."},
            {},
        ],
    )
    def test_invalid_pairs_return_none(self, raw):
        assert prefixed_pair_to_dialogue(raw, SOURCE) is None

    def test_pipeline_explodes_rows_and_reports_raw_rows(self):
        rows = list(read_jsonl(FIXTURES_DIR / "raw_prefixed_pairs.jsonl"))
        records, report = run_pipeline(
            [SourceBatch(source_tag="src:pp", format="prefixed_pairs", records=rows)],
            RULES,
            KEYWORD_MAP,
        )
        stats = report["sources"]["src:pp"]
        assert stats["raw_rows"] == 4
        assert stats["input"] == 4  # exploded pairs
        assert stats["output"] == len(records) == 4
        assert all(r.n_turns == 1 for r in records)
        assert all(validate_dialogue(r) == [] for r in records)


class TestCleanDialogue:
    def test_clean_record_passes_through_unchanged_content(self):
        rec = make_record(
            ("user", "Hi, I got your call about switching energy providers."),
            ("assistant", "Thanks for calling back! How much electricity do you use monthly?"),
        )
        result = clean_dialogue(rec, RULES)
        assert result.drop_reason is None
        assert result.pii_replacements == 0
        assert [m.content for m in result.record.messages] == [m.content for m in rec.messages]
        assert validate_dialogue(result.record) == []

    # -- rule 1: empty / too-short turns ---------------------------------
    def test_short_turn_removal_remerges_neighbours(self):
        rec = make_record(
            ("user", "Tell me about your plans."),
            ("assistant", "We offer fixed-rate plans."),
            ("user", "k"),
            ("assistant", "And variable-rate plans too."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.drop_reason is None
        assert [m.role for m in result.record.messages] == ["user", "assistant"]
        assert (
            result.record.messages[1].content
            == "We offer fixed-rate plans.\nAnd variable-rate plans too."
        )
        assert result.record.n_turns == 1

    def test_dialogue_with_no_surviving_assistant_turn_is_dropped(self):
        rec = make_record(("user", "Hello, is anyone there?"), ("assistant", "y"))
        result = clean_dialogue(rec, RULES)
        assert result.record is None
        assert result.drop_reason == DROP_EMPTY

    def test_whitespace_only_turn_is_removed(self):
        rec = make_record(
            ("user", "What is a fixed-rate plan?"),
            ("assistant", "   "),
            ("user", "Hello? Are you still there?"),
            ("assistant", "Sorry about that. A fixed-rate plan locks in your unit price."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.drop_reason is None
        assert [m.role for m in result.record.messages] == ["user", "assistant"]
        assert result.record.messages[0].content.startswith("What is a fixed-rate plan?")

    # -- rule 2: non-English ----------------------------------------------
    def test_non_english_dialogue_is_dropped(self):
        rec = make_record(
            ("user", "Können Sie mir etwas über Ihre Stromtarife erzählen?"),
            ("assistant", "Selbstverständlich, wir bieten verschiedene Tarife für Privatkunden."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.record is None
        assert result.drop_reason == DROP_NON_ENGLISH

    def test_english_system_prompt_does_not_rescue_non_english_body(self):
        rec = make_record(
            ("system", "You are a professional energy sales agent. Reply in English."),
            ("user", "Hola, ¿me puede explicar las tarifas de electricidad que ofrecen ustedes?"),
            ("assistant", "Claro que sí, ofrecemos tarifas fijas y variables para hogares."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.record is None
        assert result.drop_reason == DROP_NON_ENGLISH

    # -- rule 3: PII placeholder substitution -----------------------------
    def test_pii_is_replaced_with_placeholders(self):
        rec = make_record(
            ("user", "Call me at 555-123-4567 or email john.doe@example.com please."),
            ("assistant", "Sure, I will call 555-123-4567 and confirm by email."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.drop_reason is None
        assert result.pii_replacements == 3
        assert "[PHONE]" in result.record.messages[0].content
        assert "[EMAIL]" in result.record.messages[0].content
        assert "555-123-4567" not in result.record.messages[1].content

    def test_innocent_numbers_are_not_redacted(self):
        rec = make_record(
            ("user", "My bill says 950 kWh and I pay about 80 a month since 2026-01-15."),
            ("assistant", "Thanks! At 950 kWh a fixed plan could work well for you."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.pii_replacements == 0
        assert result.record.messages[0].content == rec.messages[0].content

    # -- rule 4: trailing non-assistant truncation -------------------------
    def test_trailing_user_turn_is_truncated(self):
        rec = make_record(
            ("user", "What plans do you offer?"),
            ("assistant", "We offer fixed and variable plans."),
            ("user", "Thanks, I will think about it. Bye."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.drop_reason is None
        assert result.record.messages[-1].role == "assistant"
        assert len(result.record.messages) == 2
        assert validate_dialogue(result.record) == []

    # -- id / n_turns recomputation ----------------------------------------
    def test_id_and_n_turns_recomputed_after_cleaning(self):
        rec = make_record(
            ("user", "What plans do you offer?"),
            ("assistant", "We offer fixed and variable plans."),
            ("user", "Thanks, bye."),
        )
        result = clean_dialogue(rec, RULES)
        assert result.record.id != "dlg-unset"
        assert result.record.id.startswith("dlg-")
        assert result.record.n_turns == 1
        # idempotent: cleaning a cleaned record changes nothing
        again = clean_dialogue(result.record, RULES)
        assert again.record == result.record


class TestRedactPii:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Reach me at (555) 123-4567.", "Reach me at [PHONE]."),
            ("My number is +1 555 123 4567 today.", "My number is [PHONE] today."),
            ("Card 4111 1111 1111 1111 on file.", "Card [CARD] on file."),
            ("Write to jane_smith+offers@mail.example.co.uk now.", "Write to [EMAIL] now."),
        ],
    )
    def test_pii_patterns_hit(self, text, expected):
        redacted, n = redact_pii(text)
        assert redacted == expected
        assert n == 1

    @pytest.mark.parametrize(
        "text",
        [
            "The plan costs $49.99 per month.",
            "Usage was 950 kWh in May.",
            "Contract ends on 2026-06-12.",
            "I pay about 80 a month.",
        ],
    )
    def test_innocent_text_untouched(self, text):
        redacted, n = redact_pii(text)
        assert redacted == text
        assert n == 0


KEYWORD_MAP = {
    "objection_handling": ["too expensive", "not interested", "already have a provider"],
    "info_gathering": ["how much electricity", "current contract", "your monthly bill"],
    "cold_open": ["is now a good time", "calling from"],
    "closing": ["sign up", "next step", "confirmation email"],
}


class TestTagScenario:
    def test_keyword_hit_assigns_scenario(self):
        rec = make_record(
            ("user", "Honestly this is too expensive for me."),
            ("assistant", "I hear you. May I ask what you pay today?"),
        )
        assert tag_scenario(rec, KEYWORD_MAP) == "objection_handling"

    def test_highest_scoring_scenario_wins(self):
        rec = make_record(
            ("user", "Is now a good time? I want to know the next step to sign up."),
            ("assistant", "Great! The next step is verification, then a confirmation email."),
        )
        # closing: "sign up" + 2x "next step" + "confirmation email" beats cold_open's 1 hit
        assert tag_scenario(rec, KEYWORD_MAP) == "closing"

    def test_no_match_falls_back_to_general(self):
        rec = make_record(
            ("user", "Can you explain what a kilowatt-hour is?"),
            ("assistant", "It is the unit your usage is measured in."),
        )
        assert tag_scenario(rec, KEYWORD_MAP) == "general"

    def test_matching_is_case_insensitive(self):
        rec = make_record(
            ("user", "I am NOT INTERESTED at all."),
            ("assistant", "Understood, thanks for your time."),
        )
        assert tag_scenario(rec, KEYWORD_MAP) == "objection_handling"

    def test_tagging_is_idempotent_and_pure(self):
        rec = make_record(
            ("user", "How much electricity do you use?"),
            ("assistant", "About 950 kWh."),
        )
        before = rec.model_copy(deep=True)
        first = tag_scenario(rec, KEYWORD_MAP)
        second = tag_scenario(rec, KEYWORD_MAP)
        assert first == second == "info_gathering"
        assert rec == before  # tag_scenario must not mutate the record

    def test_empty_keyword_map_yields_general(self):
        rec = make_record(("user", "Hello."), ("assistant", "Hi there!"))
        assert tag_scenario(rec, {}) == "general"


class TestRunPipeline:
    def _fixture_batches(self) -> list[SourceBatch]:
        return [
            SourceBatch(
                source_tag="local:alpaca-fixture",
                format="alpaca",
                records=list(read_jsonl(FIXTURES_DIR / "raw_alpaca.jsonl")),
            ),
            SourceBatch(
                source_tag="local:sharegpt-fixture",
                format="sharegpt",
                records=list(read_jsonl(FIXTURES_DIR / "raw_sharegpt.jsonl")),
            ),
        ]

    def test_fixture_sources_produce_expected_counts(self):
        records, report = run_pipeline(self._fixture_batches(), RULES, KEYWORD_MAP)

        alpaca = report["sources"]["local:alpaca-fixture"]
        assert alpaca["input"] == 9
        assert alpaca["dropped"]["conversion_failed"] == 4
        assert alpaca["dropped"]["non_english"] == 1
        assert alpaca["output"] == 4

        sharegpt = report["sources"]["local:sharegpt-fixture"]
        assert sharegpt["input"] == 10
        assert sharegpt["dropped"]["conversion_failed"] == 5
        assert sharegpt["dropped"]["validation_failed"] == 1  # gpt-first dialogue
        assert sharegpt["output"] == 4

        assert report["totals"]["input"] == 19
        assert report["totals"]["output"] == len(records) == 8
        # the PII fixture row: phone+email in both user and assistant turns
        assert report["pii_replacements"] == 4
        assert sum(report["scenario_distribution"].values()) == 8

        assert all(validate_dialogue(r) == [] for r in records)
        assert all(r.lang == "en" for r in records)

    def test_cross_source_duplicates_are_counted(self):
        raw = {
            "instruction": "Explain a fixed-rate plan.",
            "output": "It locks in your unit price for the contract length.",
        }
        batches = [
            SourceBatch(source_tag="src:a", format="alpaca", records=[raw, dict(raw)]),
            SourceBatch(source_tag="src:b", format="alpaca", records=[dict(raw)]),
        ]
        records, report = run_pipeline(batches, RULES, KEYWORD_MAP)
        assert len(records) == 1
        assert records[0].source == "src:a"  # first occurrence wins
        assert report["sources"]["src:a"]["dropped"]["duplicate"] == 1
        assert report["sources"]["src:b"]["dropped"]["duplicate"] == 1
        assert report["totals"]["dropped"]["duplicate"] == 2

    def test_unknown_format_raises(self):
        batch = SourceBatch(source_tag="src:x", format="csv", records=[])
        with pytest.raises(ValueError, match="unknown source format"):
            run_pipeline([batch], RULES, KEYWORD_MAP)

    def test_scenario_tags_use_keyword_map(self):
        raw = {
            "conversations": [
                {"from": "human", "value": "Honestly, this is too expensive for me."},
                {"from": "gpt", "value": "I understand. May I ask what you pay today?"},
            ]
        }
        records, _ = run_pipeline(
            [SourceBatch(source_tag="src:a", format="sharegpt", records=[raw])],
            RULES,
            KEYWORD_MAP,
        )
        assert records[0].scenario == "objection_handling"
