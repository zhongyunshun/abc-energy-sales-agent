"""Chat-template rendering and completion-only masking boundaries for SFT/DPO.

Pure logic only: this module deliberately imports NO ``torch`` / ``transformers``
/ ``trl`` so its unit tests run on a CPU-only host (design doc section 4.1).

Why this exists
---------------
M4 trains Qwen3-4B-Instruct with Unsloth QLoRA and **completion-only masking**
(loss on assistant turns only). The masking is the correctness core (design doc
section 6, open item 1). The locked stack is TRL 0.23.0 / Unsloth 2025.11.1, and
the chosen path is Unsloth ``train_on_responses_only`` driven by the Qwen ChatML
marker strings :data:`INSTRUCTION_PART` / :data:`RESPONSE_PART` (see report). That
path is token-based at train time, but its *boundary semantics* are reproduced
here at the character level so they can be unit-tested without a tokenizer:

- everything before the first ``<|im_start|>assistant\\n`` is masked;
- each assistant turn's content (plus its closing ``<|im_end|>\\n``) is in loss;
- user / system turns between assistant turns are masked.

The real tokenizer renders the chat template at train time; :func:`render_chatml`
mirrors the Qwen ChatML format so the marker-based boundary logic tested here is
faithful to what ``train_on_responses_only`` does on the tokenized text.
"""

from __future__ import annotations

from sales_agent.common.schema import DialogueRecord, Message, PreferencePair

# --- Qwen ChatML markers ----------------------------------------------------
# These are also the exact strings handed to Unsloth ``train_on_responses_only``
# (``instruction_part`` / ``response_part``). Keep them identical to the Qwen
# chat template; changing them silently breaks completion-only masking.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def render_chatml(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """Render messages to Qwen ChatML text.

    Each message becomes ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. When
    ``add_generation_prompt`` is true, a trailing ``<|im_start|>assistant\\n`` is
    appended (the inference prompt). This mirrors the Qwen3-Instruct chat
    template (no injected ``<think>`` block for the Instruct-2507 variant).
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{IM_START}{msg.role}\n{msg.content}{IM_END}\n")
    if add_generation_prompt:
        parts.append(RESPONSE_PART)
    return "".join(parts)


def to_conversation(record: DialogueRecord, default_system: str | None = None) -> dict:
    """Project a DialogueRecord to the conversational row SFTTrainer consumes.

    Returns ``{"messages": [{"role", "content"}, ...]}`` with only role/content
    kept (no ids/meta). When ``default_system`` is set and the record has no
    system message, it is injected at position 0; otherwise messages pass
    through unchanged (faithful to the data: M1 real records have no system,
    M2 synthetic records usually do).
    """
    msgs = [{"role": m.role, "content": m.content} for m in record.messages]
    if default_system is not None and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": default_system}, *msgs]
    return {"messages": msgs}


def completion_only_spans(
    text: str,
    response_part: str = RESPONSE_PART,
    instruction_part: str = INSTRUCTION_PART,
) -> list[tuple[int, int]]:
    """Character spans that contribute to loss under completion-only masking.

    Reproduces ``train_on_responses_only`` semantics at the character level:
    after each ``response_part`` marker, the span runs to the next
    ``instruction_part`` marker (start of the following user turn) or to the end
    of the text for the final assistant turn. Each returned span therefore
    covers the assistant content *plus* its trailing ``<|im_end|>\\n`` (so the
    model learns to emit the end-of-turn token). Everything outside the returned
    spans -- system, user, and the assistant header markers themselves -- is
    masked.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        r = text.find(response_part, i)
        if r == -1:
            break
        start = r + len(response_part)
        nxt = text.find(instruction_part, start)
        end = nxt if nxt != -1 else len(text)
        if end > start:  # skip empty spans (e.g. a trailing generation prompt)
            spans.append((start, end))
        i = max(end, start + 1)
    return spans


def assistant_content_spans(
    text: str,
    response_part: str = RESPONSE_PART,
    im_end: str = IM_END,
) -> list[tuple[int, int]]:
    """Character spans of assistant *content* only (excluding the ``<|im_end|>``).

    Unlike :func:`completion_only_spans` (which includes the closing
    ``<|im_end|>\\n`` because the model must learn to emit it), this returns just
    the answer text -- useful for asserting exact assistant-turn boundaries in
    tests. For each ``response_part`` marker the span runs to the next
    ``<|im_end|>`` (or end of text if absent).
    """
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        r = text.find(response_part, i)
        if r == -1:
            break
        start = r + len(response_part)
        e = text.find(im_end, start)
        end = e if e != -1 else len(text)
        spans.append((start, end))
        i = end + len(im_end)
    return spans


# --- M5 DPO preference-pair conversion (pure addition; M4 functions above are
#     untouched) -------------------------------------------------------------


def preference_pair_to_dpo(pair: PreferencePair, default_system: str | None = None) -> dict:
    """Convert a :class:`PreferencePair` to TRL's DPO *standard* row format.

    Returns ``{"prompt": str, "chosen": str, "rejected": str}`` where:

    - ``prompt`` is the ``context`` rendered to Qwen ChatML text with a trailing
      assistant generation prompt (``add_generation_prompt=True``), reusing
      :func:`render_chatml` so the train-time string is exactly the
      unit-tested template. Concatenated with a completion it yields
      ``...<|im_start|>assistant\\n{reply}`` -- the point the model continues from.
    - ``chosen`` / ``rejected`` are the raw assistant reply strings (no markers),
      per the M5 contract (design doc section 3-M5).

    When ``default_system`` is set and the context has no system message, it is
    injected at position 0 (mirrors :func:`to_conversation`, keeping the DPO prompt
    distribution consistent with SFT).

    Raises ``ValueError`` if ``context`` is empty or does not end with a ``user``
    message. The :class:`PreferencePair` schema already enforces this at
    construction, but the conversion layer re-checks defensively so a hand-built
    or ``model_construct``-bypassed pair can never silently emit a malformed prompt
    (design doc section 3-M5 explicitly requires this guard).
    """
    context = pair.context
    if not context:
        raise ValueError(f"{pair.id}: context must not be empty")
    if context[-1].role != "user":
        raise ValueError(
            f"{pair.id}: context must end with a user message, got {context[-1].role}"
        )
    messages = list(context)
    if default_system is not None and messages[0].role != "system":
        messages = [Message(role="system", content=default_system), *messages]
    return {
        "prompt": render_chatml(messages, add_generation_prompt=True),
        "chosen": pair.chosen,
        "rejected": pair.rejected,
    }
