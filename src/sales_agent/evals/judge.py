"""LLM-as-a-Judge scoring for the three model groups (design doc 3-M10).

Pure logic + async orchestration over an injected :class:`OpenRouterClient`:
sample selection (the SAME ids across all groups), Referee-prompt construction,
robust judge-response parsing, and cross-tab aggregation. File I/O, env loading,
and client construction live in the thin CLI (``scripts/eval/run_judge.py``).

Module boundaries (design doc 1.1): M10 consumes M9's ``results.jsonl`` rows by
their file contract only -- it reads each row's ``id`` / ``scenario`` /
``prompt_messages`` / ``completion`` (already reasoning-stripped by M8's parser +
M9's :func:`samples.strip_reasoning`; M10 never re-reads the raw model output).
Nothing here imports or mutates M9's rules/samples or ``openrouter.py``.

Judge model: must be NON-Google. The synthesizer used ``google/gemini-2.5-flash``
(risk board 2026-06-13 row 39) and the design's default judge ``gemini-2.5-pro`` is
also Google -- same-source bias. ``configs/eval_judge.yaml`` therefore defaults
``judge_models`` to non-Google models (Anthropic + OpenAI), which are a different
family from BOTH the synthesizer and the Qwen models under test (base/SFT/DPO), so
the judge favors no group. The list is config-driven (cross-validate with several
judges, or run one). The judge is BLIND -- it never sees the model_tag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2

from sales_agent.common.openrouter import OpenRouterClient, OpenRouterError, UsageStats

logger = logging.getLogger(__name__)

# Bump when editing configs/prompts/judge.j2; recorded in the run manifest.
JUDGE_TEMPLATE_VERSION = "v1"

# Four scoring dimensions (design doc 3-M10 / contract 2.4). ``hallucination`` is
# the "hallucination-free" axis: 5 = invents nothing, 1 = fabricates a price/fact.
DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "coherence",
    "sales_logic",
    "professionalism",
    "hallucination",
)
DEFAULT_SCORE_MIN = 1
DEFAULT_SCORE_MAX = 5
# Overall mean gap below which two groups are reported "no significant difference".
DEFAULT_NO_DIFF_THRESHOLD = 0.3

# Short system message; the rubric + JSON shape live in the .j2 template.
JUDGE_SYSTEM = (
    "You are a strict, impartial sales-quality evaluator. You always respond with "
    "exactly one valid JSON object and nothing else: no markdown fences, no "
    "explanation, no trailing text."
)

_ROLE_LABELS = {"system": "System", "user": "Customer", "assistant": "Agent"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    """Parsed eval_judge.yaml (design doc 3-M10 config notes)."""

    seed: int
    judge_models: tuple[str, ...]  # NON-Google; >=1 (cross-validation)
    n_samples: int  # per group (default 100); same ids across groups
    temperature: float
    max_tokens: int
    max_retries: int  # parse-failure retries per (sample, judge) -- design: <= 2
    client_max_retries: int  # transient API retries inside OpenRouterClient
    concurrency: int
    smoke_n: int
    dimensions: tuple[str, ...]
    score_min: int
    score_max: int
    no_diff_threshold: float
    use_json_mode: bool
    pricing: dict  # per-model {input_per_m, output_per_m} for cost estimate

    @classmethod
    def from_dict(cls, cfg: dict) -> JudgeConfig:
        models = cfg.get("judge_models")
        if not models:
            raise ValueError("eval_judge.yaml must set a non-empty 'judge_models' list")
        return cls(
            seed=cfg.get("seed", 42),
            judge_models=tuple(models),
            n_samples=cfg.get("sampling", {}).get("n_samples", 100),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 512),
            max_retries=cfg.get("max_retries", 2),
            client_max_retries=cfg.get("client_max_retries", 3),
            concurrency=cfg.get("concurrency", 8),
            smoke_n=cfg.get("smoke", {}).get("n_samples", 5),
            dimensions=tuple(cfg.get("dimensions") or DEFAULT_DIMENSIONS),
            score_min=cfg.get("score_min", DEFAULT_SCORE_MIN),
            score_max=cfg.get("score_max", DEFAULT_SCORE_MAX),
            no_diff_threshold=cfg.get("no_diff_threshold", DEFAULT_NO_DIFF_THRESHOLD),
            use_json_mode=cfg.get("use_json_mode", False),
            pricing=cfg.get("pricing", {}),
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeScore:
    """One judge's scores for one (sample, model_tag) under one judge model."""

    id: str
    model_tag: str
    scenario: str
    scores: dict[str, int]
    rationale: dict[str, str]
    judge_model: str
    judge_raw: str

    def to_row(self) -> dict:
        """Serialize to the M10 scores.jsonl contract (design doc 2.4)."""
        return {
            "id": self.id,
            "model_tag": self.model_tag,
            "scenario": self.scenario,
            "scores": dict(self.scores),
            "rationale": dict(self.rationale),
            "judge_model": self.judge_model,
            "judge_raw": self.judge_raw,
        }


@dataclass(frozen=True)
class JudgeParseError:
    """A rejected judge response. ``kind`` is one of: ``not_json``,
    ``missing_field``, ``out_of_range``, ``bad_type``, ``api_error``."""

    kind: str
    detail: str = ""


# ---------------------------------------------------------------------------
# T10.2 sample selection -- the SAME ids across all groups
# ---------------------------------------------------------------------------


def _stratified_ids(pairs: list[tuple[str, str]], n: int | None, seed: int) -> list[str]:
    """Pick ``n`` ids from ``(id, scenario)`` pairs, scenario-stratified by seed.

    Mirrors the proven apportionment in ``evals.samples.select_samples`` (kept a
    separate copy: modules interact by contract, not import -- design 1.1):
    largest-remainder quota per scenario, a per-scenario seeded RNG
    (``Random(f"{seed}:{sc}")``), returned grouped by sorted scenario then original
    order. Depends ONLY on (pairs, n, seed) -- never on the model under test.
    """
    if n is None or n >= len(pairs):
        return [pid for pid, _ in pairs]
    if n <= 0:
        return []

    by_sc: dict[str, list[str]] = {}
    for pid, sc in pairs:
        by_sc.setdefault(sc, []).append(pid)

    total = len(pairs)
    exact = {sc: n * len(ids) / total for sc, ids in by_sc.items()}
    quotas = {sc: int(v) for sc, v in exact.items()}
    remainder = n - sum(quotas.values())
    for sc in sorted(by_sc, key=lambda s: (-(exact[s] - quotas[s]), s))[:remainder]:
        quotas[sc] += 1

    chosen: list[str] = []
    for sc in sorted(by_sc):
        ids = by_sc[sc]
        k = min(quotas[sc], len(ids))
        rng = random.Random(f"{seed}:{sc}")
        picked = set(rng.sample(range(len(ids)), k))
        chosen.extend(ids[i] for i in range(len(ids)) if i in picked)
    return chosen


def select_judge_samples(
    results_by_tag: dict[str, list[dict]], n: int | None, seed: int
) -> dict[str, list[dict]]:
    """Select the SAME ``n`` ids for every model_tag, scenario-stratified by seed.

    M9 produces base/sft/dpo ``results.jsonl`` over the identical 650 ids; this draws
    one comparable sub-batch so the three groups are scored on exactly the same
    dialogues (design doc 3-M10). Steps: take the common id universe (intersection
    across tags), stratify-sample ``n`` from it using the FIRST tag's order as the
    canonical (id, scenario) source, then return each tag's matching rows in the
    same id order. The returned id set is therefore identical across all tags --
    asserted before return.

    Raises ``ValueError`` if ``results_by_tag`` is empty, any tag is empty, rows lack
    an ``id``, or the tags share no common ids.
    """
    if not results_by_tag:
        raise ValueError("results_by_tag is empty")

    id_maps: dict[str, dict[str, dict]] = {}
    for tag, rows in results_by_tag.items():
        if not rows:
            raise ValueError(f"tag {tag!r} has no result rows")
        m: dict[str, dict] = {}
        for r in rows:
            rid = r.get("id")
            if not rid:
                raise ValueError(f"tag {tag!r} has a row without an 'id'")
            m[rid] = r
        id_maps[tag] = m

    common: set[str] = set.intersection(*(set(m) for m in id_maps.values()))
    if not common:
        raise ValueError("tags share no common ids -- cannot compare the same samples")

    canonical_tag = next(iter(results_by_tag))
    canonical_pairs = [
        (r["id"], r.get("scenario", "general"))
        for r in results_by_tag[canonical_tag]
        if r["id"] in common
    ]
    chosen_ids = _stratified_ids(canonical_pairs, n, seed)

    selected = {tag: [id_maps[tag][rid] for rid in chosen_ids] for tag in results_by_tag}

    # The DoD guarantee: identical id set across every group.
    id_sets = [{r["id"] for r in rows} for rows in selected.values()]
    assert all(s == id_sets[0] for s in id_sets), "selected id sets diverge across tags"
    return selected


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def load_template(path: str | Path) -> jinja2.Template:
    """Load the judge Jinja2 template (whitespace trimming on, no autoescape)."""
    text = Path(path).read_text(encoding="utf-8")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)
    return env.from_string(text)


def format_dialogue(prompt_messages: list[dict]) -> str:
    """Render the prompt context as a readable [Role] transcript for the judge."""
    lines = []
    for m in prompt_messages:
        label = _ROLE_LABELS.get(m.get("role", ""), m.get("role", "?").title())
        lines.append(f"[{label}] {m.get('content', '').strip()}")
    return "\n".join(lines)


def build_judge_prompt(sample_row: dict, template: jinja2.Template) -> list[dict]:
    """Render one M9 result row into chat messages for the judge (BLIND to model_tag)."""
    user = template.render(
        dialogue=format_dialogue(sample_row.get("prompt_messages", [])),
        candidate=(sample_row.get("completion") or "").strip(),
        scenario=sample_row.get("scenario", "general"),
    ).strip()
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# T10.3a parse
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9]*\s*\n")
_FENCE_CLOSE_RE = re.compile(r"\n```\s*$")


def _extract_json(text: str) -> dict | None:
    """Decode one JSON object from judge output, tolerating a markdown fence."""
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


def _coerce_score(value: Any) -> int | None:
    """Return an int score for an int / whole-float input, else None (bad type)."""
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def parse_judge_response(
    raw: str,
    dimensions: Iterable[str] = DEFAULT_DIMENSIONS,
    score_min: int = DEFAULT_SCORE_MIN,
    score_max: int = DEFAULT_SCORE_MAX,
) -> dict | JudgeParseError:
    """Parse + validate one judge response into ``{scores, rationale}`` or an error.

    Robust to the failure modes the design calls out (3-M10 test points): non-JSON
    output, a missing dimension, a non-1-5 / non-integer score. Each dimension's
    value may be either ``{"score": int, "reason": str}`` or a bare integer score
    (reason then defaults to ""). One bad response yields a :class:`JudgeParseError`
    for the caller to retry/count -- it never raises, so a single malformed reply
    cannot crash the run.
    """
    data = _extract_json(raw)
    if data is None:
        return JudgeParseError("not_json", "no JSON object found in output")

    scores: dict[str, int] = {}
    rationale: dict[str, str] = {}
    for dim in dimensions:
        if dim not in data:
            return JudgeParseError("missing_field", f"missing dimension {dim!r}")
        entry = data[dim]
        if isinstance(entry, dict):
            raw_score = entry.get("score")
            reason = entry.get("reason", "")
        else:
            raw_score = entry
            reason = ""
        score = _coerce_score(raw_score)
        if score is None:
            return JudgeParseError("bad_type", f"{dim} score not an integer: {raw_score!r}")
        if not (score_min <= score <= score_max):
            return JudgeParseError(
                "out_of_range", f"{dim} score {score} outside [{score_min}, {score_max}]"
            )
        scores[dim] = score
        rationale[dim] = str(reason).strip()
    return {"scores": scores, "rationale": rationale}


# ---------------------------------------------------------------------------
# T10.3b aggregation -- model_tag x scenario cross tabs per judge
# ---------------------------------------------------------------------------


def _mean_std(values: list[int]) -> dict:
    """Mean + population std (std=0 for a single value) + n, rounded."""
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _cell(rows: list[JudgeScore], dimensions: tuple[str, ...]) -> dict:
    """Per-dimension {mean, std, n} for a group of scores."""
    out: dict[str, dict] = {}
    for dim in dimensions:
        vals = [r.scores[dim] for r in rows if dim in r.scores]
        if vals:
            out[dim] = _mean_std(vals)
    return out


def _pairwise_overall(
    overall: dict[str, dict], tags: list[str], dimensions: tuple[str, ...], threshold: float
) -> dict[str, list[dict]]:
    """Per-dimension pairwise overall-mean diffs, flagging gaps < threshold.

    "无明显差异" / no significant difference when |mean_a - mean_b| < threshold
    (design doc 3-M10: 0.3). Pairs follow the tag order given (base, sft, dpo).
    """
    notes: dict[str, list[dict]] = {}
    for dim in dimensions:
        dim_notes = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = tags[i], tags[j]
                if dim in overall.get(a, {}) and dim in overall.get(b, {}):
                    diff = overall[a][dim]["mean"] - overall[b][dim]["mean"]
                    dim_notes.append(
                        {
                            "a": a,
                            "b": b,
                            "diff": round(diff, 3),
                            "no_diff": abs(diff) < threshold,
                        }
                    )
        notes[dim] = dim_notes
    return notes


def aggregate_scores(
    scores: list[JudgeScore],
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    no_diff_threshold: float = DEFAULT_NO_DIFF_THRESHOLD,
) -> dict:
    """Aggregate judge scores into per-judge model_tag x scenario cross tables.

    Scores from different judge models live on different scales, so each judge is
    aggregated separately. For each judge model: an ``overall`` table (model_tag ->
    dim -> {mean, std, n}), a ``by_scenario`` breakdown, and ``pairwise`` overall-mean
    diffs flagged with ``no_diff`` (< ``no_diff_threshold``). ``model_tags`` is in a
    stable canonical order (base, sft, dpo first, then any others sorted).
    """
    tag_order = _ordered_tags({s.model_tag for s in scores})
    scenarios = sorted({s.scenario for s in scores})
    judge_models = sorted({s.judge_model for s in scores})

    judges: dict[str, dict] = {}
    for jm in judge_models:
        jrows = [s for s in scores if s.judge_model == jm]
        by_tag: dict[str, list[JudgeScore]] = {}
        for s in jrows:
            by_tag.setdefault(s.model_tag, []).append(s)
        overall = {t: _cell(by_tag[t], dimensions) for t in tag_order if t in by_tag}

        by_scenario: dict[str, dict] = {}
        for sc in scenarios:
            sc_rows = [s for s in jrows if s.scenario == sc]
            if not sc_rows:
                continue
            sc_by_tag: dict[str, list[JudgeScore]] = {}
            for s in sc_rows:
                sc_by_tag.setdefault(s.model_tag, []).append(s)
            by_scenario[sc] = {
                t: _cell(sc_by_tag[t], dimensions) for t in tag_order if t in sc_by_tag
            }

        judges[jm] = {
            "n": len(jrows),
            "overall": overall,
            "by_scenario": by_scenario,
            "pairwise": _pairwise_overall(overall, tag_order, dimensions, no_diff_threshold),
        }

    return {
        "dimensions": list(dimensions),
        "model_tags": tag_order,
        "scenarios": scenarios,
        "judge_models": judge_models,
        "no_diff_threshold": no_diff_threshold,
        "n_scores": len(scores),
        "judges": judges,
    }


def _ordered_tags(tags: set[str]) -> list[str]:
    """Canonical order: base, sft, dpo first (when present), then the rest sorted."""
    preferred = ["base", "sft", "dpo"]
    ordered = [t for t in preferred if t in tags]
    ordered.extend(sorted(tags - set(ordered)))
    return ordered


# ---------------------------------------------------------------------------
# T10.4 async orchestration
# ---------------------------------------------------------------------------


@dataclass
class JudgeRun:
    """Outcome of one ``run_judge`` call."""

    scores: list[JudgeScore]
    attempted: int
    succeeded: int
    parse_failures: int
    failures_by_kind: dict[str, int]
    validation_retries: int
    tokens_by_model: dict[str, dict]  # judge_model -> {requests, prompt_tokens, completion_tokens}
    usage: UsageStats


async def _judge_one(
    row: dict,
    model_tag: str,
    judge_model: str,
    client: OpenRouterClient,
    cfg: JudgeConfig,
    template: jinja2.Template,
    chat_kwargs: dict,
) -> tuple[JudgeScore | JudgeParseError, int, tuple[int, int]]:
    """Judge one (sample, model_tag) under one judge model, retrying parse failures.

    Returns (JudgeScore | final JudgeParseError, retries_used, (prompt_tok, completion_tok)).
    Up to ``cfg.max_retries`` re-asks on a malformed/unparseable response; a final
    failure is returned (not raised) so one bad reply can't crash the run.
    """
    messages = build_judge_prompt(row, template)
    last_error: JudgeParseError = JudgeParseError("api_error", "no attempt made")
    prompt_tok = completion_tok = 0
    for attempt in range(cfg.max_retries + 1):
        try:
            result = await client.chat(messages, model=judge_model, **chat_kwargs)
        except OpenRouterError as e:
            last_error = JudgeParseError("api_error", str(e))
            continue
        prompt_tok += result.prompt_tokens
        completion_tok += result.completion_tokens
        parsed = parse_judge_response(result.content, cfg.dimensions, cfg.score_min, cfg.score_max)
        if not isinstance(parsed, JudgeParseError):
            score = JudgeScore(
                id=row["id"],
                model_tag=model_tag,
                scenario=row.get("scenario", "general"),
                scores=parsed["scores"],
                rationale=parsed["rationale"],
                judge_model=judge_model,
                judge_raw=result.content,
            )
            return score, attempt, (prompt_tok, completion_tok)
        last_error = parsed
    return last_error, cfg.max_retries, (prompt_tok, completion_tok)


async def run_judge(
    samples_by_tag: dict[str, list[dict]],
    client: OpenRouterClient,
    cfg: JudgeConfig,
    template: jinja2.Template,
) -> JudgeRun:
    """Score every (sample, model_tag) with every configured judge model.

    Cross-validation: each judge model in ``cfg.judge_models`` scores the full
    sample set independently. Concurrency is bounded by the client's semaphore.
    Per-model token totals are accumulated from each ``ChatResult`` (for the cost
    estimate) since ``UsageStats`` aggregates across models.
    """
    chat_kwargs: dict = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    if cfg.use_json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}

    jobs: list[tuple[dict, str, str]] = []
    for judge_model in cfg.judge_models:
        for model_tag, rows in samples_by_tag.items():
            for row in rows:
                jobs.append((row, model_tag, judge_model))

    outcomes = await asyncio.gather(
        *(
            _judge_one(row, tag, jm, client, cfg, template, chat_kwargs)
            for row, tag, jm in jobs
        )
    )

    scores: list[JudgeScore] = []
    failures_by_kind: dict[str, int] = {}
    validation_retries = 0
    tokens_by_model: dict[str, dict] = {
        jm: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for jm in cfg.judge_models
    }
    for (outcome, retries_used, (ptok, ctok)), (_, _, judge_model) in zip(
        outcomes, jobs, strict=True
    ):
        validation_retries += retries_used
        bucket = tokens_by_model[judge_model]
        bucket["requests"] += 1
        bucket["prompt_tokens"] += ptok
        bucket["completion_tokens"] += ctok
        if isinstance(outcome, JudgeParseError):
            failures_by_kind[outcome.kind] = failures_by_kind.get(outcome.kind, 0) + 1
        else:
            scores.append(outcome)

    parse_failures = sum(failures_by_kind.values())
    return JudgeRun(
        scores=scores,
        attempted=len(jobs),
        succeeded=len(scores),
        parse_failures=parse_failures,
        failures_by_kind=failures_by_kind,
        validation_retries=validation_retries,
        tokens_by_model=tokens_by_model,
        usage=client.usage,
    )


# ---------------------------------------------------------------------------
# Cost + comparison.md
# ---------------------------------------------------------------------------


def estimate_cost(tokens_by_model: dict[str, dict], pricing: dict) -> dict:
    """Estimate USD per judge model from token totals + config pricing.

    ``pricing`` maps a model id to ``{input_per_m, output_per_m}`` (USD per 1M
    tokens). A model with no pricing entry contributes 0 and is flagged so the
    report/manifest shows the gap rather than silently undercounting.
    """
    per_model: dict[str, dict] = {}
    total = 0.0
    missing: list[str] = []
    for model, tok in tokens_by_model.items():
        rate = pricing.get(model)
        if not rate:
            missing.append(model)
            per_model[model] = {**tok, "usd": None}
            continue
        usd = (
            tok["prompt_tokens"] / 1_000_000 * rate["input_per_m"]
            + tok["completion_tokens"] / 1_000_000 * rate["output_per_m"]
        )
        per_model[model] = {**tok, "usd": round(usd, 4)}
        total += usd
    return {"per_model": per_model, "total_usd": round(total, 4), "missing_pricing": missing}


def _fmt(cell: dict, dim: str) -> str:
    if dim not in cell:
        return "-"
    c = cell[dim]
    return f"{c['mean']:.2f}±{c['std']:.2f}"


def render_comparison_md(
    table: dict,
    run: JudgeRun,
    cfg: JudgeConfig,
    *,
    cost: dict,
    n_per_group: int,
    sample_ids: list[str],
) -> str:
    """Render comparison.md: per-judge four-dimension base/sft/dpo tables + cost.

    Core report material for M12 (design doc 5.3). Each judge gets an overall table
    (dim x model_tag, mean±std), a per-scenario breakdown, and the pairwise
    "no significant difference" notes (gap < threshold).
    """
    tags = table["model_tags"]
    dims = table["dimensions"]
    lines: list[str] = []
    lines.append("# M10 LLM-as-a-Judge — three-group comparison")
    lines.append("")
    lines.append(
        f"Blind scoring of base vs SFT vs SFT+DPO replies (M9 `results.jsonl`, the "
        f"same {n_per_group} ids per group, scenario-stratified, seed={cfg.seed}). "
        f"Each judge sees only the dialogue + candidate reply, never the model tag. "
        f"Four dimensions, 1–5 (5 best); `hallucination` is hallucination-free "
        f"(5 = invents no price/fact). Cells are mean±std."
    )
    lines.append("")
    lines.append(f"- judges (non-Google, cross-validation): {', '.join(cfg.judge_models)}")
    lines.append(f"- samples scored per group: {n_per_group}")
    lines.append(f"- scores collected: {run.succeeded} / {run.attempted} attempted")
    if run.parse_failures:
        lines.append(f"- parse failures: {run.parse_failures} {run.failures_by_kind}")
    lines.append(f'- no-significant-difference threshold: mean gap < {cfg.no_diff_threshold}')
    lines.append("")

    for jm in table["judge_models"]:
        j = table["judges"][jm]
        lines.append(f"## Judge: `{jm}` (n={j['n']})")
        lines.append("")
        lines.append("### Overall (mean±std, higher is better)")
        lines.append("")
        lines.append("| dimension | " + " | ".join(tags) + " |")
        lines.append("|" + "---|" * (len(tags) + 1))
        for dim in dims:
            row = [_fmt(j["overall"].get(t, {}), dim) for t in tags]
            lines.append(f"| {dim} | " + " | ".join(row) + " |")
        lines.append("")

        # Pairwise significance notes.
        notes = []
        for dim in dims:
            for p in j["pairwise"].get(dim, []):
                tail = " → no significant difference" if p["no_diff"] else ""
                notes.append(f"  - {dim}: {p['a']} vs {p['b']} Δ={p['diff']:+.2f}{tail}")
        if notes:
            lines.append("Pairwise overall-mean differences:")
            lines.extend(notes)
            lines.append("")

        lines.append("### By scenario (mean per dimension)")
        lines.append("")
        for sc in table["scenarios"]:
            if sc not in j["by_scenario"]:
                continue
            lines.append(f"**{sc}**")
            lines.append("")
            lines.append("| dimension | " + " | ".join(tags) + " |")
            lines.append("|" + "---|" * (len(tags) + 1))
            for dim in dims:
                row = [_fmt(j["by_scenario"][sc].get(t, {}), dim) for t in tags]
                lines.append(f"| {dim} | " + " | ".join(row) + " |")
            lines.append("")

    lines.append("## Judge cost")
    lines.append("")
    lines.append("| judge model | requests | prompt tok | completion tok | est. USD |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, c in cost["per_model"].items():
        usd = "n/a" if c["usd"] is None else f"${c['usd']:.4f}"
        lines.append(
            f"| {model} | {c['requests']} | {c['prompt_tokens']} | "
            f"{c['completion_tokens']} | {usd} |"
        )
    lines.append(f"| **total** | | | | **${cost['total_usd']:.4f}** |")
    if cost["missing_pricing"]:
        lines.append("")
        missing = ", ".join(cost["missing_pricing"])
        lines.append(f"> No pricing configured for: {missing} (USD shown as n/a).")
    lines.append("")
    return "\n".join(lines)
