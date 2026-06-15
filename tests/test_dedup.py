"""Unit tests for exact dedup / content hashing (src/sales_agent/data/dedup.py)."""

from __future__ import annotations

from sales_agent.common.schema import DialogueRecord, Message
from sales_agent.data.dedup import content_hash, dedup_exact, dialogue_id, normalize_text


def make_record(rec_id: str, *turns: tuple[str, str]) -> DialogueRecord:
    msgs = [Message(role=r, content=c) for r, c in turns]
    return DialogueRecord(
        id=rec_id,
        source="test:source",
        scenario="general",
        lang="en",
        n_turns=sum(1 for m in msgs if m.role == "assistant"),
        messages=msgs,
    )


class TestNormalizeText:
    def test_case_and_whitespace_are_canonicalized(self):
        a = [Message(role="user", content="Hello   THERE"), Message(role="assistant", content="Hi")]
        b = [Message(role="user", content="hello there "), Message(role="assistant", content="hi")]
        assert normalize_text(a) == normalize_text(b)
        assert content_hash(a) == content_hash(b)

    def test_role_is_part_of_identity(self):
        a = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
        b = [Message(role="user", content="hello"), Message(role="user", content="hi")]
        assert content_hash(a) != content_hash(b)

    def test_dialogue_id_format(self):
        msgs = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
        rec_id = dialogue_id(msgs)
        assert rec_id == f"dlg-{content_hash(msgs)[:12]}"


class TestDedupExact:
    def test_exact_duplicates_are_dropped_keeping_first(self):
        a = make_record("dlg-a", ("user", "Hi."), ("assistant", "Hello!"))
        b = make_record("dlg-b", ("user", "hi. "), ("assistant", "HELLO!"))  # same after normalize
        c = make_record("dlg-c", ("user", "Different."), ("assistant", "Reply."))
        kept, dropped = dedup_exact([a, b, c])
        assert [r.id for r in kept] == ["dlg-a", "dlg-c"]
        assert dropped == 1

    def test_dedup_is_idempotent(self):
        records = [
            make_record("dlg-a", ("user", "Hi."), ("assistant", "Hello!")),
            make_record("dlg-b", ("user", "Hi."), ("assistant", "Hello!")),
            make_record("dlg-c", ("user", "Other."), ("assistant", "Sure.")),
        ]
        once, dropped_once = dedup_exact(records)
        twice, dropped_twice = dedup_exact(once)
        assert twice == once
        assert dropped_once == 1
        assert dropped_twice == 0

    def test_empty_input(self):
        kept, dropped = dedup_exact([])
        assert kept == []
        assert dropped == 0
