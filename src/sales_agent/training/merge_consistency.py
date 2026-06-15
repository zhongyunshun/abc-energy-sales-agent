"""M6 merge consistency comparator (T6.1) + fixed-prompt loading (T6.3 logic).

Pure logic only: this module imports NO ``torch`` / ``transformers`` / ``peft`` so
its unit tests run on a CPU-only host (design doc section 4.1). The GPU generation
lives in ``scripts/training/merge_adapter.py``; everything testable without a GPU
-- the prompt rendering and the before/after comparison -- lives here.

What it checks (design doc section 3-M6)
----------------------------------------
After ``peft merge_and_unload`` folds the DPO LoRA adapter into dense BF16 weights,
the merged model must behave identically to ``base + adapter`` (PEFT inference).
We greedy-generate the same 8 fixed prompts on both and compare prompt by prompt.

Two match modes:

- ``exact``: the decoded continuations must be byte-for-byte identical.
- ``prefix_tokens``: only the first ``prefix_n`` generated token ids must match
  -- the documented relaxation for benign BF16-merge numerical drift, which can
  flip a single late token even though the merge is mathematically ``W + s·BA``.

A merge that changes behaviour yields ``consistent == False``; the script then
prints :meth:`ConsistencyResult.render_diffs`, exits 2, and records the verdict in
the merged model's ``manifest.json`` (so the DoD "一致性检查通过记录在 manifest"
holds whether the check passes or fails).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import Message
from sales_agent.training.formatting import render_chatml

MODE_EXACT = "exact"
MODE_PREFIX_TOKENS = "prefix_tokens"
_MODES = (MODE_EXACT, MODE_PREFIX_TOKENS)
DEFAULT_PREFIX_N = 64


# --- fixed consistency prompts (T6.3) ---------------------------------------


@dataclass(frozen=True)
class ConsistencyPrompt:
    """A fixed prompt for the merge consistency check.

    ``rendered`` is the Qwen ChatML inference string (context + trailing assistant
    generation prompt) ready to feed both models, built via the shared
    :func:`formatting.render_chatml` so it matches the train/inference template.
    """

    id: str
    rendered: str


def load_consistency_prompts(
    path: str, default_system: str | None = None
) -> list[ConsistencyPrompt]:
    """Load + render the fixed consistency prompts from a JSONL fixture.

    Each line is ``{"id": str, "context": [{"role", "content"}, ...]}`` where
    ``context`` ends with a ``user`` message (the turn both models must continue).
    Reuses :func:`render_chatml` (``add_generation_prompt=True``) so the prompt is
    exactly the unit-tested template; optionally injects ``default_system`` when the
    context has no system message (keeps the prompt distribution aligned with
    training). Raises ``ValueError`` on an empty or non-user-ending context so a
    bad fixture fails loudly instead of producing a meaningless prompt.
    """
    prompts: list[ConsistencyPrompt] = []
    for raw in read_jsonl(path):
        context = [Message(**m) for m in raw["context"]]
        if not context:
            raise ValueError(f"{raw.get('id')}: consistency prompt context must not be empty")
        if context[-1].role != "user":
            raise ValueError(
                f"{raw.get('id')}: context must end with a user message, "
                f"got {context[-1].role}"
            )
        messages = list(context)
        if default_system is not None and messages[0].role != "system":
            messages = [Message(role="system", content=default_system), *messages]
        prompts.append(
            ConsistencyPrompt(
                id=raw["id"],
                rendered=render_chatml(messages, add_generation_prompt=True),
            )
        )
    return prompts


# --- comparator (T6.1) ------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """One model's greedy continuation for a single consistency prompt.

    ``text`` is the decoded continuation (``skip_special_tokens=True``); used by
    ``exact`` mode. ``token_ids`` are the *newly generated* token ids (the prompt
    excluded); used by ``prefix_tokens`` mode.
    """

    prompt_id: str
    text: str
    token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class PromptVerdict:
    """Per-prompt comparison outcome; ``detail`` holds the diff text on mismatch."""

    prompt_id: str
    match: bool
    mode: str
    detail: str = ""


@dataclass(frozen=True)
class ConsistencyResult:
    """Aggregate verdict over all consistency prompts."""

    mode: str
    prefix_n: int
    verdicts: list[PromptVerdict]

    @property
    def consistent(self) -> bool:
        """True iff every prompt matched (the gate for exit code 0)."""
        return all(v.match for v in self.verdicts)

    @property
    def n_total(self) -> int:
        return len(self.verdicts)

    @property
    def n_mismatch(self) -> int:
        return sum(not v.match for v in self.verdicts)

    @property
    def mismatched_ids(self) -> list[str]:
        return [v.prompt_id for v in self.verdicts if not v.match]

    def summary(self) -> dict:
        """Compact dict recorded under the manifest's ``consistency_check`` key."""
        return {
            "mode": self.mode,
            "prefix_n": self.prefix_n if self.mode == MODE_PREFIX_TOKENS else None,
            "n_total": self.n_total,
            "n_mismatch": self.n_mismatch,
            "consistent": self.consistent,
            "mismatched_ids": self.mismatched_ids,
        }

    def render_diffs(self) -> str:
        """Human-readable diff report for the mismatched prompts (printed on exit 2)."""
        header = (
            f"# Merge consistency FAILED: {self.n_mismatch}/{self.n_total} prompts differ "
            f"(mode={self.mode}"
            + (f", prefix_n={self.prefix_n}" if self.mode == MODE_PREFIX_TOKENS else "")
            + ")"
        )
        parts = [v.detail for v in self.verdicts if not v.match]
        return "\n\n".join([header, *parts])


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing element, or first length overrun; None if equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _first_char_divergence(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _render_one_diff(peft: Generation, merged: Generation, mode: str, prefix_n: int) -> str:
    lines = [f"### {peft.prompt_id} — MISMATCH ({mode})"]
    if mode == MODE_PREFIX_TOKENS:
        a, b = peft.token_ids[:prefix_n], merged.token_ids[:prefix_n]
        idx = _first_divergence(a, b)
        lines.append(f"first divergence at token {idx} (within first {prefix_n})")
        lo = max(0, (idx or 0) - 3)
        lines.append(f"peft   ids[{lo}:]: {a[lo : (idx or 0) + 4]}")
        lines.append(f"merged ids[{lo}:]: {b[lo : (idx or 0) + 4]}")
    else:
        idx = _first_char_divergence(peft.text, merged.text)
        lines.append(f"first divergence at char {idx}")
    lines.append(f"peft   text: {peft.text!r}")
    lines.append(f"merged text: {merged.text!r}")
    return "\n".join(lines)


def compare_one(
    peft: Generation,
    merged: Generation,
    *,
    mode: str = MODE_EXACT,
    prefix_n: int = DEFAULT_PREFIX_N,
) -> PromptVerdict:
    """Compare one prompt's PEFT vs merged generation under ``mode``.

    ``exact`` compares decoded text; ``prefix_tokens`` compares the first
    ``prefix_n`` generated token ids. Raises ``ValueError`` on an unknown mode or
    if the two generations are for different prompts (an ordering bug that would
    silently compare mismatched pairs).
    """
    if mode not in _MODES:
        raise ValueError(f"unknown match mode {mode!r}; expected one of {_MODES}")
    if peft.prompt_id != merged.prompt_id:
        raise ValueError(
            f"prompt id mismatch (generations out of order): "
            f"{peft.prompt_id!r} vs {merged.prompt_id!r}"
        )
    if mode == MODE_EXACT:
        match = peft.text == merged.text
    else:  # prefix_tokens
        match = peft.token_ids[:prefix_n] == merged.token_ids[:prefix_n]
    detail = "" if match else _render_one_diff(peft, merged, mode, prefix_n)
    return PromptVerdict(prompt_id=peft.prompt_id, match=match, mode=mode, detail=detail)


def render_consistency_report(
    result: ConsistencyResult,
    merged_gens: list[Generation],
    *,
    meta: dict | None = None,
) -> str:
    """Render the consistency check into a committable markdown evidence report.

    Unlike the gitignored merged-model dir, this report is written under
    ``reports/training/`` (committed) so the merge is auditable from the repo --
    mirroring how M4/M5 commit their training manifests. For each fixed prompt it
    records the PASS/FAIL verdict and the actual greedy continuation (on a PASS the
    output is identical on ``base+adapter`` and merged, so the merged text is shown
    once; on a FAIL the per-prompt diff is shown). ``meta`` key/values (base model,
    adapter, dtype, device, size, git commit, timestamp) are printed as a provenance
    line. ``merged_gens`` must align with ``result.verdicts``.
    """
    by_id = {g.prompt_id: g for g in merged_gens}
    verdict = "PASS" if result.consistent else "FAIL"
    extra = f", prefix_n={result.prefix_n}" if result.mode == MODE_PREFIX_TOKENS else ""
    lines: list[str] = [
        f"# M6 merge consistency check — {result.n_total} prompts (greedy)\n",
        f"**{verdict}** — {result.n_total - result.n_mismatch}/{result.n_total} prompts "
        f"match (mode={result.mode}{extra}).\n",
        "Each fixed prompt is greedy-generated on `base + DPO adapter` (PEFT inference) "
        "and on the merged dense model; the continuations are compared. A PASS means "
        "the merge is behaviour-preserving.\n",
    ]
    if meta:
        lines.append("| " + " | ".join(f"{k}: {v}" for k, v in meta.items()) + " |\n")
    for v in result.verdicts:
        flag = "PASS" if v.match else "FAIL"
        lines.append(f"\n## {v.prompt_id} — {flag} ({v.mode})\n")
        if v.match:
            mg = by_id.get(v.prompt_id)
            lines.append(
                "**Output (identical on base+adapter and merged):**\n\n"
                f"{mg.text if mg else ''}\n"
            )
        else:
            lines.append(f"```\n{v.detail}\n```\n")
    return "\n".join(lines)


def compare_generations(
    peft_gens: list[Generation],
    merged_gens: list[Generation],
    *,
    mode: str = MODE_EXACT,
    prefix_n: int = DEFAULT_PREFIX_N,
) -> ConsistencyResult:
    """Compare aligned PEFT vs merged generations prompt by prompt.

    The two lists must be the same length and in the same prompt order (enforced
    per-pair by :func:`compare_one`). Raises ``ValueError`` on a length mismatch so
    a truncated generation pass can never silently drop prompts from the check.
    """
    if len(peft_gens) != len(merged_gens):
        raise ValueError(
            f"generation count mismatch: {len(peft_gens)} peft vs {len(merged_gens)} merged"
        )
    verdicts = [
        compare_one(p, m, mode=mode, prefix_n=prefix_n)
        for p, m in zip(peft_gens, merged_gens, strict=True)
    ]
    return ConsistencyResult(mode=mode, prefix_n=prefix_n, verdicts=verdicts)
