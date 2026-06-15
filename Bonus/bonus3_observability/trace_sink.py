"""Bonus 3b observability helpers.

This module is intentionally self-contained under ``Bonus/bonus3_observability``.
It can report traces to Langfuse when the SDK and credentials are available, and
it always writes a local JSONL/HTML export so the demo remains reproducible on a
PC with no cloud keys.
"""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for trace records."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def new_run_id(prefix: str = "bonus3") -> str:
    """Create a short run id that is stable enough for file names and trace ids."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class TraceTurn:
    """One observed assistant turn in a multi-turn sales conversation."""

    run_id: str
    conversation_id: str
    turn_index: int
    scenario: str
    model: str
    endpoint: str
    prompt_messages: list[dict[str, str]]
    content: str
    scored_content: str
    reasoning_content: str | None
    latency_ms: float
    rule_flags: dict[str, bool]
    n_tokens: int
    usage: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    @property
    def trace_id(self) -> str:
        """Langfuse trace id shared by all turns in one conversation."""

        return f"{self.run_id}:{self.conversation_id}"

    @property
    def generation_id(self) -> str:
        """Langfuse generation id for this single turn."""

        return f"{self.trace_id}:turn-{self.turn_index}"

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""

        return asdict(self)


class TraceSink(Protocol):
    """Minimal sink interface used by the demo and optional eval hook."""

    name: str

    def record_turn(self, turn: TraceTurn) -> None:
        """Record one assistant turn."""

    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Flush and return sink-specific artifact paths/status."""


class NullTraceSink:
    """No-op sink for disabled instrumentation."""

    name = "disabled"

    def record_turn(self, turn: TraceTurn) -> None:
        del turn

    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        del summary
        return {"sink": self.name}


class LocalTraceSink:
    """Local JSONL plus HTML export sink."""

    name = "local"

    def __init__(self, out_dir: Path, run_id: str, project: str) -> None:
        self.out_dir = out_dir
        self.run_id = run_id
        self.project = project
        self.trace_dir = out_dir / "traces"
        self.export_dir = out_dir / "exports"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / f"{run_id}.jsonl"
        self.summary_path = self.trace_dir / f"{run_id}_summary.json"
        self.html_path = self.export_dir / f"{run_id}_trace_report.html"
        self.turns: list[TraceTurn] = []

    def record_turn(self, turn: TraceTurn) -> None:
        self.turns.append(turn)
        with self.trace_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(turn.to_json(), ensure_ascii=False) + "\n")

    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "project": self.project,
            "sink": self.name,
            "run_id": self.run_id,
            "trace_path": str(self.trace_path),
            "html_path": str(self.html_path),
            **summary,
        }
        with self.summary_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        self._write_html(summary)
        return {
            "sink": self.name,
            "trace_path": str(self.trace_path),
            "summary_path": str(self.summary_path),
            "html_path": str(self.html_path),
            "n_turns": len(self.turns),
        }

    def _write_html(self, summary: dict[str, Any]) -> None:
        """Write a compact single-file trace viewer for local review."""

        by_conversation: dict[str, list[TraceTurn]] = {}
        for turn in self.turns:
            by_conversation.setdefault(turn.conversation_id, []).append(turn)

        summary_html = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
        sections = []
        for conv_id, turns in by_conversation.items():
            cards = []
            for turn in turns:
                flagged = any(turn.rule_flags.values())
                failure_label = "failure" if flagged else "pass"
                reason = turn.reasoning_content
                if reason is None:
                    reason = "(none returned by the endpoint)"
                cards.append(
                    f"""
                    <article class="turn {failure_label}">
                      <header>
                        <h3>Turn {turn.turn_index} - {html.escape(turn.scenario)}</h3>
                        <span>{html.escape(failure_label.upper())}</span>
                      </header>
                      <dl>
                        <dt>Latency</dt><dd>{turn.latency_ms:.1f} ms</dd>
                        <dt>Rule flags</dt><dd><code>{html.escape(json.dumps(turn.rule_flags))}</code></dd>
                        <dt>Failure annotation</dt><dd><code>{html.escape(json.dumps(turn.failure, ensure_ascii=False))}</code></dd>
                      </dl>
                      <details open>
                        <summary>Model content</summary>
                        <pre>{html.escape(turn.content)}</pre>
                      </details>
                      <details>
                        <summary>Reasoning content</summary>
                        <pre>{html.escape(reason)}</pre>
                      </details>
                      <details>
                        <summary>Prompt messages</summary>
                        <pre>{html.escape(json.dumps(turn.prompt_messages, indent=2, ensure_ascii=False))}</pre>
                      </details>
                    </article>
                    """
                )
            sections.append(
                f"""
                <section>
                  <h2>{html.escape(conv_id)}</h2>
                  {''.join(cards)}
                </section>
                """
            )

        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bonus 3 Observability Trace - {html.escape(self.run_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --pass: #e9f7ef;
      --fail: #fff1f0;
      --accent: #2952a3;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #f7f8fa;
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 12px;
    }}
    h2 {{
      font-size: 21px;
      margin: 30px 0 14px;
    }}
    h3 {{
      font-size: 16px;
    }}
    .summary, .turn {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 12px 0;
    }}
    .turn.pass {{
      border-left: 6px solid #2e7d32;
      background: var(--pass);
    }}
    .turn.failure {{
      border-left: 6px solid #c62828;
      background: var(--fail);
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }}
    header span {{
      font-size: 12px;
      font-weight: 700;
      color: var(--accent);
    }}
    dl {{
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 6px 14px;
      margin: 0 0 12px;
    }}
    dt {{
      color: var(--muted);
      font-weight: 700;
    }}
    dd {{
      margin: 0;
      min-width: 0;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin: 8px 0 0;
      font-size: 13px;
    }}
    code {{
      overflow-wrap: anywhere;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <h1>Bonus 3 Observability Trace</h1>
    <p>Local export for run <code>{html.escape(self.run_id)}</code>.</p>
    <section class="summary">
      <h2>Run Summary</h2>
      <pre>{summary_html}</pre>
    </section>
    {''.join(sections)}
  </main>
</body>
</html>
"""
        self.html_path.write_text(page, encoding="utf-8", newline="\n")


class LangfuseTraceSink:
    """Best-effort Langfuse sink.

    The local sink is still used in parallel. Any SDK/runtime issue is captured in
    ``errors`` and does not fail the demo.
    """

    name = "langfuse"

    def __init__(self, project: str) -> None:
        self.project = project
        self.errors: list[str] = []
        try:
            from langfuse import Langfuse  # type: ignore

            self.client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST"),
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary.
            self.client = None
            self.available = False
            self.errors.append(f"Langfuse init failed: {exc}")

    def record_turn(self, turn: TraceTurn) -> None:
        if not self.available or self.client is None:
            return
        try:
            # This shape matches the Langfuse v2 SDK and is harmlessly skipped if
            # a future SDK changes the API.
            self.client.trace(
                id=turn.trace_id,
                name=f"{self.project}:{turn.conversation_id}",
                input=turn.prompt_messages,
                metadata={
                    "run_id": turn.run_id,
                    "conversation_id": turn.conversation_id,
                    "scenario": turn.scenario,
                    "endpoint": turn.endpoint,
                },
                tags=["bonus3", "reasoning-trace", turn.scenario],
            )
            self.client.generation(
                id=turn.generation_id,
                trace_id=turn.trace_id,
                name=f"turn-{turn.turn_index}",
                model=turn.model,
                input=turn.prompt_messages,
                output={
                    "content": turn.content,
                    "reasoning_content": turn.reasoning_content,
                },
                metadata={
                    "latency_ms": round(turn.latency_ms, 3),
                    "rule_flags": turn.rule_flags,
                    "failure": turn.failure,
                    "n_tokens": turn.n_tokens,
                },
                usage=turn.usage or None,
            )
        except Exception as exc:  # noqa: BLE001 - observability must not break eval/demo.
            self.errors.append(f"Langfuse record failed for {turn.generation_id}: {exc}")
            self.available = False

    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        del summary
        if self.available and self.client is not None:
            try:
                self.client.flush()
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"Langfuse flush failed: {exc}")
        return {
            "sink": self.name,
            "available": self.available,
            "errors": self.errors,
        }


class CompositeTraceSink:
    """Fan out one turn to several sinks."""

    def __init__(self, sinks: list[TraceSink]) -> None:
        self.sinks = sinks
        self.name = "+".join(s.name for s in sinks)
        self.run_id = getattr(sinks[0], "run_id", None) if sinks else None

    def record_turn(self, turn: TraceTurn) -> None:
        for sink in self.sinks:
            sink.record_turn(turn)

    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "sink": self.name,
            "children": [sink.finalize(summary) for sink in self.sinks],
        }


def has_langfuse_env() -> bool:
    """Return True when the minimum Langfuse credential set is present."""

    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def make_trace_sink(
    *,
    out_dir: Path,
    run_id: str,
    project: str,
    provider: str = "auto",
) -> TraceSink:
    """Build the requested trace sink.

    ``provider`` may be ``auto``, ``local``, ``langfuse``, or ``disabled``. Auto
    mode uses Langfuse only when credentials are present; local export is always
    enabled except for ``disabled``.
    """

    provider = provider.lower()
    if provider == "disabled":
        return NullTraceSink()

    local = LocalTraceSink(out_dir=out_dir, run_id=run_id, project=project)
    if provider == "local":
        return local

    should_try_langfuse = provider == "langfuse" or (provider == "auto" and has_langfuse_env())
    if should_try_langfuse:
        langfuse = LangfuseTraceSink(project=project)
        return CompositeTraceSink([local, langfuse])

    return local


class Timer:
    """Tiny latency helper for endpoint calls."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = getattr(self, "end", time.perf_counter())
        return (end - self.start) * 1000.0
