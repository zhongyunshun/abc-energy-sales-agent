"""Eval-sample construction and reasoning-prefix stripping (the M9 contract, T9.1).

Pure logic only (no torch / transformers / network) so it unit-tests on the
CPU-only Windows host. Two responsibilities:

1. :func:`build_eval_samples` turns each ``test.jsonl`` dialogue into one
   :class:`EvalSample`: the context up to (but not including) the final assistant
   turn becomes the prompt; that final assistant turn becomes the ``gold``
   reference. Every other module reads/writes the shared ``DialogueRecord``
   contract (the dialogue contract), but M9 only needs id / scenario / messages, so this
   accepts plain dicts (the thin CLI feeds it ``read_jsonl`` rows) and validates
   the minimum it relies on -- raising on malformed input rather than silently
   dropping records.
2. :func:`strip_reasoning` defensively removes a *leading* ``<think>...</think>``
   block. M8's vLLM serve uses the qwen3 reasoning parser so endpoint
   ``message.content`` is already clean (the empty ``<think></think>`` the Qwen3
   chat template injects is moved to ``reasoning_content``); but the local /
   base-model inference path (T9.4) has no such parser, so the scoring pipeline
   strips again here. This keeps the over_length token count and role_break check
   from being polluted by an empty (or non-empty) think prefix -- pinned by unit
   tests as required by the 2026-06-14 Option A decision.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# Strip a SINGLE leading <think>...</think> block (possibly empty), tolerating
# surrounding whitespace. DOTALL so the block can span newlines; only anchored at
# the start (\A) so a think tag appearing mid-reply is left untouched.
_LEADING_THINK_RE = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove one leading ``<think>...</think>`` block, then strip whitespace.

    Returns the answer text the model actually emitted for the user. An empty
    think (``<think>\\n\\n</think>``), a non-empty think, or no think at all are
    all handled; a think tag that is not at the very start is preserved.
    """
    return _LEADING_THINK_RE.sub("", text, count=1).strip()


@dataclass(frozen=True)
class EvalSample:
    """One offline-eval unit: a prompt to generate from + the gold reference.

    - ``prompt_messages``: the dialogue context (role/content dicts) ending with
      the user turn the model must answer -- everything before the final
      assistant turn (so earlier assistant turns are kept for multi-turn context).
    - ``gold``: the reference final assistant reply (used for optional PPL and for
      M10 judge context; rule metrics score the *generated* reply, not gold).
    """

    id: str
    scenario: str
    prompt_messages: list[dict]
    gold: str
    meta: dict = field(default_factory=dict)


def _to_sample(rec: dict) -> EvalSample:
    """Build one EvalSample from a dialogue dict, raising ValueError if malformed.

    A record must have a non-empty ``messages`` list ending with an assistant
    turn, and the resulting prompt must be non-empty and end with a user turn.
    """
    rid = rec.get("id", "<no-id>")
    messages = rec.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{rid}: messages missing or empty")
    last = messages[-1]
    if last.get("role") != "assistant":
        raise ValueError(f"{rid}: last message is {last.get('role')!r}, expected assistant")
    gold = last.get("content") or ""
    if not gold.strip():
        raise ValueError(f"{rid}: final assistant content is empty")
    prompt = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
    if not prompt or prompt[-1]["role"] != "user":
        tail = prompt[-1]["role"] if prompt else "<empty>"
        raise ValueError(f"{rid}: prompt must end with a user turn, got {tail}")
    return EvalSample(
        id=rid,
        scenario=rec.get("scenario", "general"),
        prompt_messages=prompt,
        gold=gold,
        meta=dict(rec.get("meta", {})),
    )


def build_eval_samples(records: Iterable[dict]) -> list[EvalSample]:
    """Construct one :class:`EvalSample` per dialogue record (the M9 sample-construction step).

    Records are the JSONL rows of the test set (``DialogueRecord`` shape). Raises
    ``ValueError`` on the first malformed record -- the contract test set ends
    every dialogue with an assistant turn, so a violation is a real input fault
    the CLI surfaces as exit code 2 rather than silently dropping samples.
    """
    return [_to_sample(rec) for rec in records]


def select_samples(samples: list[EvalSample], n: int | None, seed: int) -> list[EvalSample]:
    """Pick ``n`` samples, stratified by scenario, deterministically by ``seed``.

    ``n`` is ``None`` or >= the pool size -> return all samples unchanged (the
    full-test-set default; the cheapest "same batch" guarantee). Otherwise allocate
    the quota across scenarios proportionally (largest-remainder), sample within
    each scenario with a per-scenario seeded RNG, and return in stable
    (scenario-sorted, original-order) order.

    The batch depends ONLY on (samples, n, seed) -- never on the model under test --
    so all three groups (base/sft/dpo) evaluated with the same config see exactly
    the same ids, and M10 can re-draw the same ids from the results (the M9/M10 sampling contract).
    """
    if n is None or n >= len(samples):
        return list(samples)
    if n <= 0:
        return []

    by_scenario: dict[str, list[EvalSample]] = {}
    for s in samples:
        by_scenario.setdefault(s.scenario, []).append(s)

    total = len(samples)
    # Largest-remainder apportionment so the per-scenario quotas sum exactly to n.
    exact = {sc: n * len(rows) / total for sc, rows in by_scenario.items()}
    quotas = {sc: int(v) for sc, v in exact.items()}
    remainder = n - sum(quotas.values())
    for sc in sorted(by_scenario, key=lambda s: (-(exact[s] - quotas[s]), s))[:remainder]:
        quotas[sc] += 1

    chosen: list[EvalSample] = []
    for sc in sorted(by_scenario):
        rows = by_scenario[sc]
        k = min(quotas[sc], len(rows))
        rng = random.Random(f"{seed}:{sc}")
        picked = set(rng.sample(range(len(rows)), k))
        chosen.extend(rows[i] for i in range(len(rows)) if i in picked)
    return chosen
