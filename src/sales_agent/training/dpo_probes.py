"""M5 behaviour-probe construction and before/after diff rendering (T5.3).

Pure logic only (no torch/transformers): the 20 fixed probe prompts that easily
elicit *pushy* closes or *rate-hallucination* are loaded and rendered here, and
the before(SFT)/after(DPO) greedy generations are laid out into the
``dpo_behavior_diff.md`` report. The GPU generation itself lives in
``scripts/training/train_dpo.py``; everything testable without a GPU lives here.

The probe contract (``tests/fixtures/dpo_probes.jsonl``): one JSON object per
line, ``{"id", "category": "pushy"|"rate_hallucination", "context": [messages]}``
where ``context`` ends with a ``user`` message (the turn the agent must answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import Message
from sales_agent.training.formatting import render_chatml

# Empty reasoning block the Qwen3-Instruct-2507 chat template can prepend to a
# generation (see M5/M4 report); stripped for readable diffs.
_THINK_PREFIX = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class Probe:
    """A behaviour probe: a prompt context plus the failure mode it targets."""

    id: str
    category: str
    context: list[Message]

    @property
    def last_user(self) -> str:
        """The final user message text (the line the agent must respond to)."""
        return next(m.content for m in reversed(self.context) if m.role == "user")


def load_probes(path: str) -> list[Probe]:
    """Parse the probe fixture, validating each context ends with a user turn.

    Raises ``ValueError`` on a malformed probe (empty context, non-user-ending
    context, or unknown role) so a bad fixture fails loudly instead of producing
    a meaningless prompt.
    """
    probes: list[Probe] = []
    for raw in read_jsonl(path):
        context = [Message(**m) for m in raw["context"]]
        if not context:
            raise ValueError(f"{raw.get('id')}: probe context must not be empty")
        if context[-1].role != "user":
            raise ValueError(
                f"{raw.get('id')}: probe context must end with a user message, "
                f"got {context[-1].role}"
            )
        probes.append(Probe(id=raw["id"], category=raw["category"], context=context))
    return probes


def build_probe_prompt(probe: Probe, default_system: str | None = None) -> str:
    """Render a probe context to a Qwen ChatML inference prompt.

    Mirrors :func:`formatting.preference_pair_to_dpo`'s prompt side: the context
    rendered with a trailing assistant generation prompt, optionally injecting a
    default system message when absent (keeps the probe distribution aligned with
    the DPO training prompts).
    """
    messages = list(probe.context)
    if default_system is not None and messages[0].role != "system":
        messages = [Message(role="system", content=default_system), *messages]
    return render_chatml(messages, add_generation_prompt=True)


def strip_think_prefix(text: str) -> str:
    """Drop a leading empty ``<think></think>`` block (Qwen3-Instruct artifact)."""
    return _THINK_PREFIX.sub("", text).strip()


def count_changed(before: list[str], after: list[str]) -> int:
    """Number of probes whose greedy continuation changed after DPO (think-stripped)."""
    return sum(
        strip_think_prefix(b) != strip_think_prefix(a)
        for b, a in zip(before, after, strict=True)
    )


def render_behavior_diff(
    probes: list[Probe],
    before: list[str],
    after: list[str],
    *,
    before_label: str = "SFT (pre-DPO)",
    after_label: str = "DPO (post)",
    meta: dict | None = None,
    conclusion: str | None = None,
) -> str:
    """Assemble the before/after greedy-generation markdown report.

    ``before`` and ``after`` are the generations (same order as ``probes``) from
    the pre-DPO (SFT) and post-DPO models. ``meta`` key/values are printed as a
    provenance line. ``conclusion`` (if given) is appended as a "## Conclusion"
    section -- the recorded read of whether visible convergence was observed.
    Raises ``ValueError`` on length mismatch so a truncated run never silently
    drops probes from the report.
    """
    if not (len(probes) == len(before) == len(after)):
        raise ValueError(
            f"length mismatch: {len(probes)} probes, {len(before)} before, {len(after)} after"
        )
    n_changed = count_changed(before, after)
    lines: list[str] = [
        f"# M5 DPO behaviour diff — {len(probes)} probes (greedy)\n",
        f"Before = **{before_label}**, after = **{after_label}**. Each probe is a "
        "context that tends to elicit a pushy close or an invented rate; we compare "
        "the greedy continuation before vs. after DPO alignment.\n",
    ]
    if meta:
        lines.append("| " + " | ".join(f"{k}: {v}" for k, v in meta.items()) + " |\n")

    by_cat: dict[str, int] = {}
    for probe in probes:
        by_cat[probe.category] = by_cat.get(probe.category, 0) + 1
    lines.append("Probe mix: " + ", ".join(f"{k} × {v}" for k, v in sorted(by_cat.items())) + "\n")
    lines.append(f"Greedy output changed on **{n_changed}/{len(probes)}** probes after DPO.\n")

    for i, probe in enumerate(probes):
        changed = strip_think_prefix(before[i]) != strip_think_prefix(after[i])
        flag = "" if changed else " — _identical_"
        lines.append(f"\n## {probe.id} ({probe.category}){flag}\n")
        lines.append(f"**User:** {probe.last_user}\n")
        lines.append(f"**{before_label}:** {strip_think_prefix(before[i])}\n")
        lines.append(f"**{after_label}:** {strip_think_prefix(after[i])}\n")

    if conclusion:
        lines.append(f"\n## Conclusion\n\n{conclusion}\n")
    return "\n".join(lines)
