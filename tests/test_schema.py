"""Tests for common/schema.py against the contract fixtures."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import DialogueRecord, PreferencePair, validate_dialogue

FIXTURES = Path(__file__).parent / "fixtures"

DIALOGUES_VALID = list(read_jsonl(FIXTURES / "dialogues_valid.jsonl"))
DIALOGUES_INVALID = list(read_jsonl(FIXTURES / "dialogues_invalid.jsonl"))
PREFS_VALID = list(read_jsonl(FIXTURES / "preference_pairs_valid.jsonl"))
PREFS_INVALID = list(read_jsonl(FIXTURES / "preference_pairs_invalid.jsonl"))


class TestDialogueRecordValid:
    @pytest.mark.parametrize("raw", DIALOGUES_VALID, ids=lambda r: r["id"])
    def test_valid_fixture_passes(self, raw):
        rec = DialogueRecord.model_validate(raw)
        assert validate_dialogue(rec) == []

    def test_fixture_coverage(self):
        """Valid fixtures must cover both system/no-system and several scenarios."""
        assert len(DIALOGUES_VALID) >= 5
        has_system = {r["messages"][0]["role"] == "system" for r in DIALOGUES_VALID}
        assert has_system == {True, False}
        scenarios = {r["scenario"] for r in DIALOGUES_VALID}
        assert {
            "objection_handling",
            "info_gathering",
            "cold_open",
            "closing",
            "general",
        } <= scenarios

    def test_meta_defaults_to_empty_dict(self):
        raw = {k: v for k, v in DIALOGUES_VALID[0].items() if k != "meta"}
        rec = DialogueRecord.model_validate(raw)
        assert rec.meta == {}

    def test_roundtrip_serialization(self):
        rec = DialogueRecord.model_validate(DIALOGUES_VALID[0])
        again = DialogueRecord.model_validate(rec.model_dump())
        assert again == rec


class TestDialogueRecordInvalid:
    STRUCTURAL = [r for r in DIALOGUES_INVALID if r["stage"] == "structural"]
    SEMANTIC = [r for r in DIALOGUES_INVALID if r["stage"] == "semantic"]

    @pytest.mark.parametrize("case", STRUCTURAL, ids=lambda c: c["reason"])
    def test_structural_rejected_by_pydantic(self, case):
        with pytest.raises(ValidationError) as exc_info:
            DialogueRecord.model_validate(case["record"])
        assert case["expect"] in str(exc_info.value)

    @pytest.mark.parametrize("case", SEMANTIC, ids=lambda c: c["reason"])
    def test_semantic_rejected_by_validate_dialogue(self, case):
        rec = DialogueRecord.model_validate(case["record"])
        errors = validate_dialogue(rec)
        assert errors, f"expected semantic errors for: {case['reason']}"
        assert any(case["expect"] in e for e in errors), (
            f"expected an error containing {case['expect']!r}, got {errors}"
        )

    def test_multiple_errors_collected(self):
        """validate_dialogue accumulates independent errors instead of stopping."""
        rec = DialogueRecord.model_validate(
            {
                "id": "dlg-multi-err",
                "source": "synthetic:v1",
                "scenario": "general",
                "lang": "zh",
                "n_turns": 5,
                "messages": [
                    {"role": "user", "content": "Hi."},
                    {"role": "assistant", "content": "Hello!"},
                ],
            }
        )
        errors = validate_dialogue(rec)
        assert any("lang" in e for e in errors)
        assert any("n_turns" in e for e in errors)


class TestPreferencePair:
    @pytest.mark.parametrize("raw", PREFS_VALID, ids=lambda r: r["id"])
    def test_valid_fixture_passes(self, raw):
        pair = PreferencePair.model_validate(raw)
        assert pair.context[-1].role == "user"
        assert pair.chosen != pair.rejected

    def test_fixture_coverage(self):
        assert len(PREFS_VALID) >= 5
        scenarios = {r["scenario"] for r in PREFS_VALID}
        assert {"pushy", "rate_hallucination"} <= scenarios

    @pytest.mark.parametrize("case", PREFS_INVALID, ids=lambda c: c["reason"])
    def test_invalid_rejected(self, case):
        with pytest.raises(ValidationError) as exc_info:
            PreferencePair.model_validate(case["record"])
        assert case["expect"] in str(exc_info.value)

    def test_meta_defaults_to_empty_dict(self):
        raw = {k: v for k, v in PREFS_VALID[0].items() if k != "meta"}
        pair = PreferencePair.model_validate(raw)
        assert pair.meta == {}
