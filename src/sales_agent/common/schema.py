"""Data contracts shared by all pipeline modules (design doc section 2).

Every module's input validation, output generation, and test fixtures use
these Pydantic models. Files on disk are JSONL (UTF-8, one object per line).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A single chat message."""

    role: Role
    content: str


class DialogueRecord(BaseModel):
    """Multi-turn dialogue contract (design doc section 2.1).

    Shared by M1 (normalize), M2 (synthesize), M3 (split), M9/M10 (eval).
    Structural checks live in the model; semantic checks (role sequence,
    turn count, language) live in :func:`validate_dialogue` so callers can
    collect per-record errors instead of a single exception.
    """

    id: str
    source: str
    scenario: str
    lang: str
    n_turns: int
    meta: dict = Field(default_factory=dict)
    messages: list[Message]


def validate_dialogue(record: DialogueRecord) -> list[str]:
    """Semantic validation for a structurally valid DialogueRecord.

    Returns a list of human-readable error strings; an empty list means the
    record is valid. Rules (design doc section 2.1):

    1. Role sequence: at most one system message and only at position 0;
       the rest must be strictly alternating user/assistant starting with
       user and ending with assistant.
    2. Every message content is non-empty (after stripping whitespace).
    3. ``n_turns`` equals the actual number of assistant messages.
    4. ``lang`` is ``"en"``.
    """
    errors: list[str] = []

    msgs = record.messages
    if not msgs:
        errors.append("messages is empty")
        return errors

    # Rule 1: role sequence
    offset = 1 if msgs[0].role == "system" else 0
    body = msgs[offset:]
    for i, msg in enumerate(body):
        if msg.role == "system":
            errors.append(f"system message at position {i + offset} (only allowed at index 0)")
            break
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.role != expected:
            errors.append(
                f"role sequence broken at body position {i}: expected {expected}, got {msg.role}"
            )
            break
    if not body:
        errors.append("dialogue has no user/assistant turns")
    elif body[-1].role != "assistant" and not any(e.startswith("role sequence") for e in errors):
        errors.append(f"dialogue must end with assistant, got {body[-1].role}")

    # Rule 2: non-empty content
    for i, msg in enumerate(msgs):
        if not msg.content.strip():
            errors.append(f"empty content at message index {i}")

    # Rule 3: n_turns consistency
    actual_turns = sum(1 for m in msgs if m.role == "assistant")
    if record.n_turns != actual_turns:
        errors.append(f"n_turns={record.n_turns} but found {actual_turns} assistant messages")

    # Rule 4: language
    if record.lang != "en":
        errors.append(f'lang must be "en", got "{record.lang}"')

    return errors


class PreferencePair(BaseModel):
    """DPO preference pair contract (design doc section 2.2).

    Produced by M2 (synthesize), consumed by M5 (DPO training).
    ``context`` must be a valid prompt prefix ending with a user message.
    """

    id: str
    scenario: str
    context: list[Message]
    chosen: str
    rejected: str
    meta: dict = Field(default_factory=dict)

    @field_validator("chosen", "rejected")
    @classmethod
    def _non_empty_response(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("response must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_context(self) -> PreferencePair:
        if not self.context:
            raise ValueError("context must not be empty")
        if self.context[-1].role != "user":
            raise ValueError(f"context must end with a user message, got {self.context[-1].role}")
        for i, msg in enumerate(self.context):
            if msg.role == "system" and i != 0:
                raise ValueError(f"system message at position {i} (only allowed at index 0)")
            if not msg.content.strip():
                raise ValueError(f"empty content at context index {i}")
        return self
