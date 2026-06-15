"""Normalize raw Alpaca/ShareGPT sales dialogues into DialogueRecord (design doc section 3-M1).

Pure logic only: format conversion, the fixed-order cleaning chain, scenario
tagging, and pipeline orchestration over already-loaded raw records. File and
dataset I/O lives in the thin CLI (``scripts/data/normalize.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from sales_agent.common.schema import DialogueRecord, Message, validate_dialogue
from sales_agent.data.dedup import content_hash, dialogue_id

# langdetect is probabilistic by default; fix its seed for reproducible runs.
DetectorFactory.seed = 42


def _build_record(messages: list[Message], source: str, raw_format: str) -> DialogueRecord:
    return DialogueRecord(
        id=dialogue_id(messages),
        source=source,
        scenario="general",
        lang="en",
        n_turns=sum(1 for m in messages if m.role == "assistant"),
        meta={"raw_format": raw_format},
        messages=messages,
    )


def alpaca_to_dialogue(raw: dict, source: str) -> DialogueRecord | None:
    """Convert one Alpaca record (instruction/input/output) to a single-turn dialogue.

    ``instruction`` (plus optional ``input``, appended after a blank line)
    becomes the user message; ``output`` becomes the assistant message.
    Returns None when ``instruction`` or ``output`` is missing, non-string,
    or blank — the caller counts these as conversion failures.
    """
    instruction = raw.get("instruction")
    output = raw.get("output")
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    if not isinstance(output, str) or not output.strip():
        return None
    user_content = instruction.strip()
    extra = raw.get("input")
    if isinstance(extra, str) and extra.strip():
        user_content = f"{user_content}\n\n{extra.strip()}"
    messages = [
        Message(role="user", content=user_content),
        Message(role="assistant", content=output.strip()),
    ]
    return _build_record(messages, source, raw_format="alpaca")


_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _merge_consecutive(messages: list[Message]) -> list[Message]:
    """Merge runs of same-role messages into one, joined by newlines."""
    merged: list[Message] = []
    for msg in messages:
        if merged and merged[-1].role == msg.role:
            merged[-1] = Message(role=msg.role, content=merged[-1].content + "\n" + msg.content)
        else:
            merged.append(msg)
    return merged


def sharegpt_to_dialogue(raw: dict, source: str) -> DialogueRecord | None:
    """Convert one ShareGPT record (``conversations`` list) to a dialogue.

    Maps ``human/gpt`` to ``user/assistant``, merges consecutive same-role
    messages, and hoists system messages to position 0 (multiple system
    messages are merged into one). Returns None when the record is missing
    the ``conversations`` list or contains an unknown role / non-string
    value. Role-order problems (e.g. a dialogue starting with assistant)
    are NOT silently repaired here; they surface in final validation.
    """
    convs = raw.get("conversations")
    if not isinstance(convs, list) or not convs:
        return None
    system_parts: list[str] = []
    body: list[Message] = []
    for item in convs:
        if not isinstance(item, dict):
            return None
        role = _SHAREGPT_ROLE_MAP.get(item.get("from"))
        value = item.get("value")
        if role is None or not isinstance(value, str):
            return None
        if role == "system":
            system_parts.append(value)
        else:
            body.append(Message(role=role, content=value))
    if not body:
        return None
    body = _merge_consecutive(body)
    messages = body
    if system_parts:
        messages = [Message(role="system", content="\n".join(system_parts)), *body]
    return _build_record(messages, source, raw_format="sharegpt")


_CUSTOMER_PREFIX = "Customer:"
_SALESMAN_PREFIX = "Salesman:"


def explode_prefixed_pairs(row: dict) -> list[dict]:
    """Split one column-per-turn row into independent customer/salesman pairs.

    Adapter for ``goendalf666/sales-conversations``-style rows: string columns
    named "0".."19" holding ``Customer: ...`` / ``Salesman: ...`` prefixed
    text, with None tails. Adjacent exchanges within a row are usually
    unrelated topic-wise (the rows are themed bundles, not coherent
    conversations), so each Customer+Salesman pair becomes its own
    single-turn dialogue; coherent multi-turn data comes from M2 synthesis.
    Cells that don't form a Customer-then-Salesman pair are skipped.
    """
    keys = sorted((k for k in row if isinstance(k, str) and k.isdigit()), key=int)
    cells = [row[k].strip() for k in keys if isinstance(row[k], str) and row[k].strip()]
    pairs: list[dict] = []
    i = 0
    while i < len(cells) - 1:
        if cells[i].startswith(_CUSTOMER_PREFIX) and cells[i + 1].startswith(_SALESMAN_PREFIX):
            pairs.append(
                {
                    "customer": cells[i][len(_CUSTOMER_PREFIX) :].strip(),
                    "salesman": cells[i + 1][len(_SALESMAN_PREFIX) :].strip(),
                }
            )
            i += 2
        else:
            i += 1
    return pairs


def prefixed_pair_to_dialogue(raw: dict, source: str) -> DialogueRecord | None:
    """Convert one exploded customer/salesman pair to a single-turn dialogue."""
    customer, salesman = raw.get("customer"), raw.get("salesman")
    if not isinstance(customer, str) or not customer.strip():
        return None
    if not isinstance(salesman, str) or not salesman.strip():
        return None
    messages = [
        Message(role="user", content=customer.strip()),
        Message(role="assistant", content=salesman.strip()),
    ]
    return _build_record(messages, source, raw_format="prefixed_pairs")


# ---------------------------------------------------------------------------
# Cleaning chain (design doc section 3-M1, step 3 — order is fixed)
# ---------------------------------------------------------------------------

DROP_EMPTY = "empty_after_filter"
DROP_NON_ENGLISH = "non_english"

# PII patterns. Card must be substituted before phone so that 13-16 digit
# card numbers are not partially consumed as phone numbers. Phone targets
# NANP-style numbers with an optional international prefix.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,15}\d(?![\d-])")
_PHONE_RE = re.compile(
    r"(?<![\w.-])(?:\+\d{1,2}[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-]?)\d{3}[ .-]?\d{4}(?![\w-])"
)
_PII_SUBSTITUTIONS = ((_EMAIL_RE, "[EMAIL]"), (_CARD_RE, "[CARD]"), (_PHONE_RE, "[PHONE]"))


@dataclass(frozen=True)
class CleanConfig:
    """Thresholds for the cleaning chain (values come from configs/normalize.yaml)."""

    min_content_chars: int = 2


@dataclass
class CleanResult:
    """Outcome of cleaning one record. ``record is None`` means dropped."""

    record: DialogueRecord | None
    drop_reason: str | None = None
    pii_replacements: int = 0


def redact_pii(text: str) -> tuple[str, int]:
    """Replace emails / card numbers / phone numbers with placeholders.

    Returns the redacted text and the number of replacements made.
    """
    total = 0
    for pattern, placeholder in _PII_SUBSTITUTIONS:
        text, n = pattern.subn(placeholder, text)
        total += n
    return text, total


def _detect_lang(messages: list[Message]) -> str:
    # System prompts are often instruction-English regardless of dialogue
    # language, so detect on user/assistant content only.
    text = " ".join(m.content for m in messages if m.role != "system")
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def clean_dialogue(rec: DialogueRecord, rules: CleanConfig) -> CleanResult:
    """Apply the fixed-order cleaning chain to one converted record.

    Order (design doc section 3-M1): (1) drop empty/too-short turns,
    (2) drop non-English dialogues, (3) replace PII with placeholders,
    (4) truncate trailing non-assistant turns. Exact dedup runs afterwards
    at pipeline level. The surviving record gets a recomputed id and
    n_turns since cleaning may have changed the content.
    """
    # (1) empty / too-short turns; removal can leave same-role neighbours,
    # so re-merge to preserve alternation where possible.
    msgs = [m for m in rec.messages if len(m.content.strip()) >= rules.min_content_chars]
    msgs = _merge_consecutive(msgs)
    if not any(m.role == "assistant" for m in msgs):
        return CleanResult(None, DROP_EMPTY)

    # (2) non-English dialogues
    if _detect_lang(msgs) != "en":
        return CleanResult(None, DROP_NON_ENGLISH)

    # (3) PII placeholder substitution
    pii_total = 0
    redacted: list[Message] = []
    for m in msgs:
        content, n = redact_pii(m.content)
        pii_total += n
        redacted.append(Message(role=m.role, content=content) if n else m)

    # (4) trailing non-assistant truncation (an assistant turn exists, see (1))
    while redacted[-1].role != "assistant":
        redacted.pop()

    cleaned = DialogueRecord(
        id=dialogue_id(redacted),
        source=rec.source,
        scenario=rec.scenario,
        lang="en",
        n_turns=sum(1 for m in redacted if m.role == "assistant"),
        meta=rec.meta,
        messages=redacted,
    )
    return CleanResult(cleaned, pii_replacements=pii_total)


def tag_scenario(rec: DialogueRecord, keyword_map: dict[str, list[str]]) -> str:
    """Keyword-rule scenario tagging; falls back to ``general``.

    Each scenario is scored by total (case-insensitive) substring hit count
    over the dialogue text. The highest-scoring scenario wins; ties resolve
    to the first scenario in ``keyword_map`` insertion order. A score of
    zero everywhere yields ``general``.
    """
    text = " ".join(m.content for m in rec.messages).lower()
    best, best_score = "general", 0
    for scenario, keywords in keyword_map.items():
        score = sum(text.count(kw.lower()) for kw in keywords)
        if score > best_score:
            best, best_score = scenario, score
    return best


# ---------------------------------------------------------------------------
# Pipeline orchestration over already-loaded raw records
# ---------------------------------------------------------------------------

DROP_CONVERSION = "conversion_failed"
DROP_DUPLICATE = "duplicate"
DROP_VALIDATION = "validation_failed"

_CONVERTERS = {
    "alpaca": alpaca_to_dialogue,
    "sharegpt": sharegpt_to_dialogue,
    "prefixed_pairs": prefixed_pair_to_dialogue,
}

KNOWN_FORMATS = tuple(_CONVERTERS)


@dataclass
class SourceBatch:
    """Raw records of one input source, already loaded from disk or HF."""

    source_tag: str
    format: str  # "alpaca" | "sharegpt"
    records: list[dict]


def run_pipeline(
    batches: list[SourceBatch],
    rules: CleanConfig,
    keyword_map: dict[str, list[str]],
) -> tuple[list[DialogueRecord], dict]:
    """Convert, clean, dedup, tag, and validate all sources (design doc section 3-M1).

    Per-record steps follow the fixed design order: format conversion ->
    cleaning chain -> exact dedup (global, first occurrence wins) ->
    scenario tagging -> final validate_dialogue() gate. Returns the
    surviving records and a report dict with per-source input/output and
    per-reason drop counts.
    """
    per_source: dict[str, dict] = {}
    cleaned: list[DialogueRecord] = []
    pii_total = 0

    for batch in batches:
        converter = _CONVERTERS.get(batch.format)
        if converter is None:
            raise ValueError(f"unknown source format {batch.format!r} for {batch.source_tag!r}")
        stats = per_source.setdefault(batch.source_tag, {"input": 0, "output": 0, "dropped": {}})
        records = batch.records
        if batch.format == "prefixed_pairs":
            # One raw row bundles several independent exchanges; the report
            # counts exploded pairs as input and keeps the row count visible.
            stats["raw_rows"] = stats.get("raw_rows", 0) + len(records)
            records = [pair for row in records for pair in explode_prefixed_pairs(row)]
        stats["input"] += len(records)
        for raw in records:
            rec = converter(raw, batch.source_tag)
            if rec is None:
                _bump(stats, DROP_CONVERSION)
                continue
            result = clean_dialogue(rec, rules)
            if result.record is None:
                _bump(stats, result.drop_reason)
                continue
            pii_total += result.pii_replacements
            cleaned.append(result.record)

    seen_hashes: set[str] = set()
    final: list[DialogueRecord] = []
    for rec in cleaned:
        stats = per_source[rec.source]
        h = content_hash(rec.messages)
        if h in seen_hashes:
            _bump(stats, DROP_DUPLICATE)
            continue
        seen_hashes.add(h)
        rec.scenario = tag_scenario(rec, keyword_map)
        errors = validate_dialogue(rec)
        if errors:
            _bump(stats, DROP_VALIDATION)
            continue
        stats["output"] += 1
        final.append(rec)

    scenario_dist: dict[str, int] = {}
    for rec in final:
        scenario_dist[rec.scenario] = scenario_dist.get(rec.scenario, 0) + 1

    totals = {"input": 0, "output": 0, "dropped": {}}
    for stats in per_source.values():
        totals["input"] += stats["input"]
        totals["output"] += stats["output"]
        for reason, n in stats["dropped"].items():
            totals["dropped"][reason] = totals["dropped"].get(reason, 0) + n

    report = {
        "sources": per_source,
        "totals": totals,
        "pii_replacements": pii_total,
        "scenario_distribution": dict(sorted(scenario_dist.items())),
    }
    return final, report


def _bump(stats: dict, reason: str) -> None:
    stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
