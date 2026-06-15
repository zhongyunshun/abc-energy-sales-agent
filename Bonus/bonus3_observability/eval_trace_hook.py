"""Optional Bonus 3b trace hook for offline evaluation.

This file is a small, opt-in adapter that future M9 runs can import without
changing the existing eval package. It is disabled by default and has no side
effects unless the caller explicitly creates a sink and calls
``record_eval_generation``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sales_agent.evals.rules import RuleConfig, apply_rules
from sales_agent.evals.samples import strip_reasoning

try:
    from .trace_sink import TraceSink, TraceTurn, make_trace_sink, new_run_id
except ImportError:  # pragma: no cover - supports direct script-path imports.
    from trace_sink import TraceSink, TraceTurn, make_trace_sink, new_run_id


def build_eval_trace_sink(
    *,
    enabled: bool = False,
    out_dir: str | Path = Path(__file__).resolve().parent,
    provider: str = "auto",
    project: str = "sales-agent-bonus3",
    run_id: str | None = None,
) -> TraceSink | None:
    """Return a trace sink when explicitly enabled, otherwise ``None``."""

    if not enabled:
        return None
    return make_trace_sink(
        out_dir=Path(out_dir),
        run_id=run_id or new_run_id("eval-trace"),
        project=project,
        provider=provider,
    )


def _maybe_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_reasoning_from_raw(raw: Any) -> str | None:
    """Best-effort extraction from an OpenAI/vLLM response object."""

    try:
        message = raw.choices[0].message
    except Exception:  # noqa: BLE001 - raw may be absent or test fake.
        return None
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        return reasoning
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get("reasoning_content")
    if hasattr(message, "model_dump"):
        data = message.model_dump()
        return data.get("reasoning_content")
    return None


def record_eval_generation(
    sink: TraceSink | None,
    *,
    sample: Any,
    output: Any,
    model: str,
    endpoint: str,
    latency_ms: float | None = None,
    rule_cfg: RuleConfig | None = None,
) -> dict[str, Any] | None:
    """Record one eval sample generation and return the computed trace payload.

    This mirrors the M9 row-building sequence: strip reasoning, apply rule flags,
    then attach those flags to the trace. The caller owns where this hook is
    invoked; M9 itself remains untouched for Bonus 3.
    """

    if sink is None:
        return None

    content = _maybe_get(output, "content", "") or ""
    scored_content = strip_reasoning(content)
    rule_cfg = rule_cfg or RuleConfig.from_dict(None)
    flags, n_tokens = apply_rules(
        scored_content,
        _maybe_get(sample, "scenario", "general"),
        rule_cfg,
    )
    failure = {
        "is_failure": any(flags.values()),
        "dimensions": [name for name, value in flags.items() if value],
    }
    turn = TraceTurn(
        run_id=getattr(sink, "run_id", "eval-trace"),
        conversation_id=_maybe_get(sample, "id", "unknown-sample"),
        turn_index=1,
        scenario=_maybe_get(sample, "scenario", "general"),
        model=model,
        endpoint=endpoint,
        prompt_messages=list(_maybe_get(sample, "prompt_messages", [])),
        content=content,
        scored_content=scored_content,
        reasoning_content=_extract_reasoning_from_raw(_maybe_get(output, "raw")),
        latency_ms=float(latency_ms or 0.0),
        rule_flags=flags,
        n_tokens=n_tokens,
        usage={
            "completion_tokens": _maybe_get(output, "usage_completion_tokens"),
        },
        failure=failure,
    )
    sink.record_turn(turn)
    return turn.to_json()
