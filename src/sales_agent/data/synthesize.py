"""Synthesize energy-sales dialogues and DPO preference pairs (design doc section 3-M2).

Pure logic only: task-matrix expansion, few-shot seed selection, prompt
construction, the generation-side quality gate (``parse_and_validate``), and the
async orchestration over an injected :class:`OpenRouterClient`. File I/O, env
loading, and client construction live in the thin CLI
(``scripts/data/synthesize.py``).

The quality gate reuses the shared contracts in ``common/schema.py``
(``DialogueRecord`` / ``PreferencePair`` + ``validate_dialogue``) so synthesized
records satisfy exactly the same rules as every other module's data.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
from pydantic import ValidationError

from sales_agent.common.io import read_jsonl
from sales_agent.common.openrouter import OpenRouterClient, OpenRouterError, UsageStats
from sales_agent.common.schema import (
    DialogueRecord,
    Message,
    PreferencePair,
    validate_dialogue,
)
from sales_agent.data.dedup import dialogue_id, normalize_text

logger = logging.getLogger(__name__)

# Bump when editing configs/prompts/synth_*.j2; recorded in every record's meta.
TEMPLATE_VERSION = "v1"
SYNTH_SOURCE = "synthetic:v1"

# Short system message prepended by build_prompt to nudge JSON-only output.
# Task-specific instructions (including "output JSON") live in the .j2 templates.
GENERATOR_SYSTEM = (
    "You are a meticulous synthetic-data generator. You always respond with "
    "exactly one valid JSON object and nothing else: no markdown fences, no "
    "explanation, no trailing text."
)

# Business rule (design doc section 3-M2 step 3): the assistant must not invent
# concrete prices/rates. These match currency-prefixed amounts and per-unit
# energy rates while leaving bare usage figures ("950 kWh") and word numbers
# alone. Extra patterns can be added via config `price_patterns`.
DEFAULT_PRICE_PATTERNS: tuple[str, ...] = (
    r"[$£€]\s?\d",                                                  # $0.09, £50, € 30
    r"\d+(?:\.\d+)?\s*(?:cents?|pence|p)\s*(?:/|per)\s*kwh",        # 9.5 cents per kWh, 12 p/kWh
    r"\d+(?:\.\d+)?\s*(?:cents?|pence)\b",                          # 9.5 cents
    r"\d+(?:\.\d+)?\s*(?:dollars?|pounds?|euros?|usd|gbp|eur)\b",   # 35 dollars
)


# ---------------------------------------------------------------------------
# Result / task / config types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthError:
    """A rejected generation, with the failing gate stage and a short detail.

    ``kind`` is one of: ``not_json``, ``schema``, ``semantic``,
    ``too_few_turns``, ``price_in_assistant``, ``price_in_chosen``,
    ``low_divergence``, ``api_error``.
    """

    kind: str
    detail: str = ""


@dataclass
class SynthTask:
    """One generation task. ``prompt_vars`` carries everything the Jinja
    template needs (including pre-serialized few-shot ``examples``)."""

    mode: str  # "dialogues" | "preferences"
    scenario: str  # DialogueRecord.scenario, or the PreferencePair failure mode
    index: int  # global unique nonce: drives seed rotation and diversity
    prompt_vars: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthConfig:
    """Parsed synthesize.yaml (design doc section 3-M2 config notes)."""

    seed: int
    model: str
    concurrency: int
    temperature: float
    max_tokens: int
    max_retries: int  # validation-failure retries per task (design: <= 2)
    client_max_retries: int  # transient API retries inside OpenRouterClient
    min_turns: int  # dialogue assistant-turn lower bound
    min_edit_distance: float  # chosen/rejected divergence lower bound (0..1)
    seed_examples_range: tuple[int, int]  # per-prompt few-shot count, sampled in [lo, hi]
    smoke_per_scenario: int
    use_json_mode: bool
    dialogues: dict
    preferences: dict
    pricing: dict
    price_patterns: tuple[re.Pattern, ...]

    @classmethod
    def from_dict(cls, cfg: dict) -> SynthConfig:
        patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_PRICE_PATTERNS]
        for extra in cfg.get("price_patterns") or []:
            patterns.append(re.compile(extra, re.IGNORECASE))
        # n_seed_examples may be an int (fixed) or a [lo, hi] list (random per task).
        n_seed = cfg.get("n_seed_examples", 1)
        seed_range = (n_seed, n_seed) if isinstance(n_seed, int) else tuple(n_seed[:2])
        return cls(
            seed=cfg.get("seed", 42),
            model=cfg["model"],
            concurrency=cfg.get("concurrency", 8),
            temperature=cfg.get("temperature", 0.9),
            max_tokens=cfg.get("max_tokens", 1500),
            max_retries=cfg.get("max_retries", 2),
            client_max_retries=cfg.get("client_max_retries", 3),
            min_turns=cfg.get("min_turns", 3),
            min_edit_distance=cfg.get("min_edit_distance", 0.3),
            seed_examples_range=seed_range,
            smoke_per_scenario=cfg.get("smoke_per_scenario", 2),
            use_json_mode=cfg.get("use_json_mode", False),
            dialogues=cfg.get("dialogues", {}),
            preferences=cfg.get("preferences", {}),
            pricing=cfg.get("pricing", {}),
            price_patterns=tuple(patterns),
        )


@dataclass
class SynthResult:
    """Outcome of one ``run_synthesis`` call."""

    records: list[dict]
    attempted: int
    succeeded: int
    abandoned: int
    validation_retries: int
    errors_by_kind: dict[str, int]
    usage: UsageStats
    abandoned_samples: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Few-shot seed examples (configs/prompts/seeds/*.jsonl)
# ---------------------------------------------------------------------------


def load_seeds(path: str | Path, key_field: str) -> dict[str, list[dict]]:
    """Load seed examples grouped by ``key_field`` (e.g. ``scenario``).

    The grouping field is stripped from each returned payload so the example
    shown to the model is exactly the JSON shape we ask it to produce.
    """
    grouped: dict[str, list[dict]] = {}
    for row in read_jsonl(path):
        key = row.get(key_field)
        payload = {k: v for k, v in row.items() if k != key_field}
        grouped.setdefault(key, []).append(payload)
    return grouped


def select_seeds(
    seeds_by_key: dict[str, list[dict]], key: str, rng: random.Random, count: int
) -> list[dict]:
    """Randomly pick up to ``count`` distinct seeds for ``key`` using ``rng``.

    The caller supplies a per-task seeded RNG (see ``_task_rng``), so selection
    is random across tasks yet fully reproducible for a given config seed.
    """
    pool = seeds_by_key.get(key) or []
    if not pool or count <= 0:
        return []
    return rng.sample(pool, min(count, len(pool)))


def _task_rng(seed: int, index: int) -> random.Random:
    """Deterministic per-task RNG so seed selection varies but reproduces."""
    return random.Random(seed * 1_000_003 + index)


def _pick_examples(
    cfg: SynthConfig, seeds_by_key: dict[str, list[dict]] | None, key: str, index: int
) -> list[str]:
    if not seeds_by_key:
        return []
    rng = _task_rng(cfg.seed, index)
    lo, hi = cfg.seed_examples_range
    count = rng.randint(lo, hi)
    return _serialize_examples(select_seeds(seeds_by_key, key, rng, count))


# ---------------------------------------------------------------------------
# T2.2 task-matrix expansion
# ---------------------------------------------------------------------------


def expand_task_matrix(
    cfg: SynthConfig,
    mode: str,
    *,
    seeds_by_key: dict[str, list[dict]] | None = None,
    per_scenario_limit: int | None = None,
) -> list[SynthTask]:
    """Expand the scenario x persona x ... matrix into per-task generation jobs.

    The full Cartesian product of the secondary dimensions is shuffled
    deterministically (``cfg.seed``) per scenario, then sampled (cycling when
    the quota exceeds the number of combinations) to hit each scenario's quota.
    ``per_scenario_limit`` caps every quota (used by ``--smoke``). ``mode`` is
    explicit because dialogues and preferences have different dimensions and
    different ``scenario`` semantics.
    """
    rng = random.Random(cfg.seed)
    if mode == "dialogues":
        return _expand_dialogues(cfg, rng, seeds_by_key, per_scenario_limit)
    if mode == "preferences":
        return _expand_preferences(cfg, rng, seeds_by_key, per_scenario_limit)
    raise ValueError(f"unknown mode {mode!r}; expected 'dialogues' or 'preferences'")


def _serialize_examples(seeds: list[dict]) -> list[str]:
    return [json.dumps(s, ensure_ascii=False) for s in seeds]


def _expand_dialogues(
    cfg: SynthConfig,
    rng: random.Random,
    seeds_by_key: dict[str, list[dict]] | None,
    per_scenario_limit: int | None,
) -> list[SynthTask]:
    spec = cfg.dialogues
    personas = spec["personas"]
    objection_types = spec["objection_types"]
    outcomes = spec["outcomes"]
    n_lo, n_hi = spec.get("n_turns_range", [cfg.min_turns, cfg.min_turns + 3])
    tasks: list[SynthTask] = []
    index = 0
    for sc in spec["scenarios"]:
        name = sc["name"]
        combos = list(itertools.product(personas, objection_types, outcomes))
        rng.shuffle(combos)
        quota = sc["quota"]
        if per_scenario_limit is not None:
            quota = min(quota, per_scenario_limit)
        for k in range(quota):
            persona, objection_type, outcome = combos[k % len(combos)]
            prompt_vars = {
                "scenario": name,
                "scenario_hint": sc.get("hint", ""),
                "scenario_directive": sc.get("directive", ""),
                "persona": persona,
                "objection_type": objection_type,
                "outcome": outcome,
                "n_turns": rng.randint(n_lo, n_hi),
                "examples": _pick_examples(cfg, seeds_by_key, name, index),
            }
            tasks.append(SynthTask("dialogues", name, index, prompt_vars))
            index += 1
    return tasks


def _expand_preferences(
    cfg: SynthConfig,
    rng: random.Random,
    seeds_by_key: dict[str, list[dict]] | None,
    per_scenario_limit: int | None,
) -> list[SynthTask]:
    spec = cfg.preferences
    personas = spec["personas"]
    ctx_scenarios = spec["context_scenarios"]
    tasks: list[SynthTask] = []
    index = 0
    for fm in spec["failure_modes"]:
        name = fm["name"]
        combos = list(itertools.product(personas, ctx_scenarios))
        rng.shuffle(combos)
        quota = fm["quota"]
        if per_scenario_limit is not None:
            quota = min(quota, per_scenario_limit)
        for k in range(quota):
            persona, ctx_sc = combos[k % len(combos)]
            prompt_vars = {
                "failure_mode": name,
                "failure_mode_desc": fm.get("desc", ""),
                "rejected_directive": fm.get("rejected_directive", ""),
                "persona": persona,
                "context_scenario": ctx_sc,
                "examples": _pick_examples(cfg, seeds_by_key, name, index),
            }
            tasks.append(SynthTask("preferences", name, index, prompt_vars))
            index += 1
    return tasks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def load_template(path: str | Path) -> jinja2.Template:
    """Load a Jinja2 template from disk with whitespace trimming enabled."""
    text = Path(path).read_text(encoding="utf-8")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)
    return env.from_string(text)


def build_prompt(task: SynthTask, template: jinja2.Template) -> list[dict]:
    """Render ``task`` into chat messages for the strong model."""
    user = template.render(**task.prompt_vars).strip()
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# T2.3 generation-side quality gate
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9]*\s*\n")
_FENCE_CLOSE_RE = re.compile(r"\n```\s*$")


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of one JSON object from model output.

    Strips a leading/trailing markdown code fence, then decodes the first
    balanced object starting at the first ``{``. Returns None when no JSON
    object can be parsed.
    """
    if not text or not text.strip():
        return None
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s)
        s = _FENCE_CLOSE_RE.sub("", s).strip()
    start = s.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def contains_price(text: str, patterns: Iterable[re.Pattern]) -> bool:
    """True if any configured price/rate pattern matches ``text``."""
    return any(p.search(text) for p in patterns)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_WS_RE = re.compile(r"\s+")


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance normalized to 0..1 (0 == identical after casefold).

    Inputs are lowercased and whitespace-collapsed first, so replies differing
    only in case/spacing are treated as identical. Used as the chosen/rejected
    divergence check: a degenerate pair where the two replies are identical or
    near-identical scores close to 0 and is rejected.
    """
    a = _WS_RE.sub(" ", a.strip().lower())
    b = _WS_RE.sub(" ", b.strip().lower())
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b))


def _messages_from(raw_list: Any) -> list[Message]:
    """Build Message objects, raising on any structurally bad item."""
    if not isinstance(raw_list, list):
        raise TypeError("expected a list of messages")
    return [Message(**m) for m in raw_list]


def parse_and_validate(
    raw_text: str, task: SynthTask, cfg: SynthConfig
) -> DialogueRecord | PreferencePair | SynthError:
    """The generation-side quality gate (design doc section 3-M2 step 3/4).

    Dialogue mode: JSON parse -> Message build -> DialogueRecord -> semantic
    ``validate_dialogue`` -> turn-count lower bound -> no price digits in any
    assistant turn. Preference mode: JSON parse -> PreferencePair (context ends
    with user, etc.) -> chosen must be price-clean -> chosen/rejected divergence
    lower bound. Any failure returns a :class:`SynthError` for retry/abandon.
    """
    data = extract_json(raw_text)
    if data is None:
        return SynthError("not_json", "no JSON object found in output")
    if task.mode == "dialogues":
        return _validate_dialogue(data, task, cfg)
    return _validate_preference(data, task, cfg)


def _meta_for(task: SynthTask, cfg: SynthConfig, keys: tuple[str, ...]) -> dict:
    meta = {"synth_model": cfg.model, "template_version": TEMPLATE_VERSION}
    for key in keys:
        if key in task.prompt_vars:
            meta[key] = task.prompt_vars[key]
    return meta


def _validate_dialogue(
    data: dict, task: SynthTask, cfg: SynthConfig
) -> DialogueRecord | SynthError:
    try:
        messages = _messages_from(data.get("messages"))
    except (TypeError, ValidationError) as e:
        return SynthError("schema", f"bad messages: {e}")
    if not messages:
        return SynthError("schema", "messages is empty")
    try:
        record = DialogueRecord(
            id=dialogue_id(messages),
            source=SYNTH_SOURCE,
            scenario=task.scenario,
            lang="en",
            n_turns=sum(1 for m in messages if m.role == "assistant"),
            meta=_meta_for(task, cfg, ("persona", "objection_type", "outcome")),
            messages=messages,
        )
    except ValidationError as e:
        return SynthError("schema", str(e))
    errors = validate_dialogue(record)
    if errors:
        return SynthError("semantic", "; ".join(errors))
    if record.n_turns < cfg.min_turns:
        return SynthError("too_few_turns", f"{record.n_turns} < {cfg.min_turns}")
    for m in record.messages:
        if m.role == "assistant" and contains_price(m.content, cfg.price_patterns):
            return SynthError("price_in_assistant", m.content[:120])
    return record


def _validate_preference(
    data: dict, task: SynthTask, cfg: SynthConfig
) -> PreferencePair | SynthError:
    chosen, rejected = data.get("chosen"), data.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return SynthError("schema", "chosen/rejected must be strings")
    try:
        context = _messages_from(data.get("context"))
    except (TypeError, ValidationError) as e:
        return SynthError("schema", f"bad context: {e}")
    try:
        pair = PreferencePair(
            id=preference_id(context, chosen, rejected),
            scenario=task.scenario,
            context=context,
            chosen=chosen,
            rejected=rejected,
            meta=_meta_for(task, cfg, ("persona", "context_scenario")),
        )
    except ValidationError as e:
        return SynthError("schema", str(e))
    if contains_price(pair.chosen, cfg.price_patterns):
        return SynthError("price_in_chosen", pair.chosen[:120])
    dist = normalized_edit_distance(pair.chosen, pair.rejected)
    if dist < cfg.min_edit_distance:
        return SynthError("low_divergence", f"{dist:.3f} < {cfg.min_edit_distance}")
    return pair


def preference_id(context: list[Message], chosen: str, rejected: str) -> str:
    """Stable content-derived id for a preference pair (``pref-`` + 12 hex)."""
    payload = (
        normalize_text(context)
        + "\nCHOSEN: "
        + chosen.strip().lower()
        + "\nREJECTED: "
        + rejected.strip().lower()
    )
    return f"pref-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# T2.4 async orchestration
# ---------------------------------------------------------------------------


async def _synthesize_one(
    task: SynthTask,
    client: OpenRouterClient,
    cfg: SynthConfig,
    template: jinja2.Template,
    chat_kwargs: dict,
) -> tuple[DialogueRecord | PreferencePair | SynthError, int]:
    """Generate one record, retrying validation failures up to cfg.max_retries.

    Returns (record-or-final-error, retries_used). Concurrency is bounded by the
    client's semaphore (one acquire per chat call).
    """
    messages = build_prompt(task, template)
    last_error: SynthError = SynthError("api_error", "no attempt made")
    for attempt in range(cfg.max_retries + 1):
        try:
            result = await client.chat(messages, **chat_kwargs)
        except OpenRouterError as e:
            last_error = SynthError("api_error", str(e))
            continue
        parsed = parse_and_validate(result.content, task, cfg)
        if not isinstance(parsed, SynthError):
            return parsed, attempt
        last_error = parsed
    return last_error, cfg.max_retries


async def run_synthesis(
    tasks: list[SynthTask],
    client: OpenRouterClient,
    cfg: SynthConfig,
    template: jinja2.Template,
) -> SynthResult:
    """Run all tasks concurrently and collect validated records + stats.

    Token usage accumulates on ``client.usage`` (used for the cost report).
    Validation-failed tasks are retried (<= cfg.max_retries) then abandoned and
    counted by error kind.
    """
    chat_kwargs: dict = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    if cfg.use_json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}

    outcomes = await asyncio.gather(
        *(_synthesize_one(t, client, cfg, template, chat_kwargs) for t in tasks)
    )

    records: list[dict] = []
    abandoned_samples: list[dict] = []
    errors_by_kind: dict[str, int] = {}
    succeeded = 0
    validation_retries = 0
    for (outcome, retries_used), task in zip(outcomes, tasks, strict=True):
        validation_retries += retries_used
        if isinstance(outcome, SynthError):
            errors_by_kind[outcome.kind] = errors_by_kind.get(outcome.kind, 0) + 1
            if len(abandoned_samples) < 20:
                abandoned_samples.append(
                    {
                        "index": task.index,
                        "scenario": task.scenario,
                        "kind": outcome.kind,
                        "detail": outcome.detail,
                    }
                )
        else:
            records.append(outcome.model_dump())
            succeeded += 1

    abandoned = len(tasks) - succeeded
    return SynthResult(
        records=records,
        attempted=len(tasks),
        succeeded=succeeded,
        abandoned=abandoned,
        validation_retries=validation_retries,
        errors_by_kind=errors_by_kind,
        usage=client.usage,
        abandoned_samples=abandoned_samples,
    )
