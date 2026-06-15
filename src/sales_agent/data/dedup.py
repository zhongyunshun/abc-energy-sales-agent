"""Exact deduplication and content hashing (M1; reused by M3 for global re-dedup).

Duplicates are detected on a normalized rendering of the whole dialogue:
lowercased content, collapsed whitespace, role-prefixed lines. The same hash
also powers stable record ids (``dlg-`` + first 12 hex chars of the SHA-256),
so id equality is equivalent to exact-duplicate equality.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from sales_agent.common.schema import DialogueRecord, Message

_WS_RE = re.compile(r"\s+")


def normalize_text(messages: list[Message]) -> str:
    """Canonical text rendering used for hashing and duplicate detection."""
    return "\n".join(f"{m.role}: {_WS_RE.sub(' ', m.content.strip().lower())}" for m in messages)


def content_hash(messages: list[Message]) -> str:
    """Hex SHA-256 of the normalized dialogue text."""
    return hashlib.sha256(normalize_text(messages).encode("utf-8")).hexdigest()


def dialogue_id(messages: list[Message]) -> str:
    """Stable content-derived record id (design doc section 2.1)."""
    return f"dlg-{content_hash(messages)[:12]}"


def dedup_exact(records: Iterable[DialogueRecord]) -> tuple[list[DialogueRecord], int]:
    """Drop exact duplicates, keeping the first occurrence per content hash.

    Returns (kept records in input order, number of dropped duplicates).
    """
    seen: set[str] = set()
    kept: list[DialogueRecord] = []
    dropped = 0
    for rec in records:
        h = content_hash(rec.messages)
        if h in seen:
            dropped += 1
        else:
            seen.add(h)
            kept.append(rec)
    return kept, dropped
