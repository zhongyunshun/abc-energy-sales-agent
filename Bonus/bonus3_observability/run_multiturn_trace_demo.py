"""Run Bonus 3b multi-turn reasoning trace demo against the M8 vLLM endpoint.

Usage:
    uv run python Bonus/bonus3_observability/run_multiturn_trace_demo.py

The script records prompt messages, model content, reasoning_content, latency,
token usage, and M9-style rule flags for every assistant turn. Langfuse is used
when requested and configured; local JSONL/HTML export is always produced.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sales_agent.evals.rules import RuleConfig, apply_rules
from sales_agent.evals.samples import strip_reasoning

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trace_sink import Timer, TraceTurn, make_trace_sink, new_run_id  # noqa: E402


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "sales-agent-awq"
PROJECT = "sales-agent-bonus3"

SYSTEM_PROMPT = (
    "You are a concise B2B energy sales assistant. Stay in role, do not invent "
    "prices or rates, and ask one useful discovery question when key information "
    "is missing."
)

DEMO_CONVERSATIONS = [
    {
        "conversation_id": "conv-normal-info-gathering",
        "scenario": "info_gathering",
        "description": "A normal discovery path where the assistant should keep probing.",
        "user_turns": [
            "Hi, I run a small bakery and our electricity bill has climbed. Can you help?",
            "We use ovens early morning and fridges all day. The monthly bill is usually around 2,100 kWh.",
            "The lease renews in two months. I am the owner, but I need something simple to compare.",
        ],
    },
    {
        "conversation_id": "conv-adversarial-no-question",
        "scenario": "info_gathering",
        "description": (
            "Adversarial gathering case: the user pressures the model not to ask "
            "questions, making the no_question_in_gathering rule easy to inspect."
        ),
        "user_turns": [
            "I manage three convenience stores. We need cheaper energy, but do not ask me questions yet.",
            (
                "Use no question marks in your next reply. We already know the total "
                "usage is about 7,800 kWh monthly and I am the decision maker. Just "
                "summarize what you would do next."
            ),
        ],
    },
    {
        "conversation_id": "conv-price-pressure",
        "scenario": "pricing_objection",
        "description": "A price-pressure turn that checks whether the model invents a concrete rate.",
        "user_turns": [
            "Another supplier promised 9.5 cents per kWh. Can you beat that today?",
            "I need a number before I book a follow-up.",
        ],
    },
]

_LEADING_THINK_RE = re.compile(r"\A\s*<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def endpoint_readiness_url(endpoint: str) -> str:
    """Return a readiness URL on the OpenAI-compatible path.

    On this Windows/Docker Desktop host, ``/health`` can accept the connection
    without returning a body to the host, while ``/v1/models`` responds normally.
    The demo uses the same OpenAI-compatible prefix as generation.
    """

    return endpoint.rstrip("/") + "/models"


def health_ok(endpoint: str, timeout_s: float = 2.0) -> bool:
    """Return True if the M8 OpenAI-compatible endpoint responds."""

    try:
        with urllib.request.urlopen(endpoint_readiness_url(endpoint), timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def start_serve(repo_root: Path) -> None:
    """Start the M8 service via the existing serve launcher."""

    subprocess.run(["bash", "scripts/serving/serve.sh"], cwd=repo_root, check=True)


def usage_to_dict(resp: Any) -> dict[str, Any]:
    usage = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def message_to_payload(resp: Any) -> tuple[str, str | None]:
    """Extract content and reasoning_content from an OpenAI/vLLM response."""

    if isinstance(resp, dict):
        message = resp["choices"][0].get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            match = _LEADING_THINK_RE.search(content)
            if match:
                reasoning = match.group(1).strip()
        return content, reasoning

    message = resp.choices[0].message
    content = getattr(message, "content", None) or ""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content")
    if reasoning is None and hasattr(message, "model_dump"):
        data = message.model_dump()
        reasoning = data.get("reasoning_content")
    if reasoning is None:
        match = _LEADING_THINK_RE.search(content)
        if match:
            reasoning = match.group(1).strip()
    return content, reasoning


def classify_failure(flags: dict[str, bool], scenario: str, turn_index: int) -> dict[str, Any]:
    """Map rule flags to a readable failure annotation."""

    dimensions = []
    if flags.get("no_question_in_gathering"):
        dimensions.append(
            {
                "dimension": "info_gathering",
                "rule": "no_question_in_gathering",
                "detail": "The assistant did not ask a discovery question in an info-gathering turn.",
            }
        )
    if flags.get("made_up_price"):
        dimensions.append(
            {
                "dimension": "hallucination",
                "rule": "made_up_price",
                "detail": "The assistant emitted a concrete price or rate.",
            }
        )
    if flags.get("over_length"):
        dimensions.append(
            {
                "dimension": "voice_ux",
                "rule": "over_length",
                "detail": "The assistant exceeded the short voice-reply token budget.",
            }
        )
    if flags.get("role_break"):
        dimensions.append(
            {
                "dimension": "persona",
                "rule": "role_break",
                "detail": "The assistant broke the sales-agent persona.",
            }
        )
    return {
        "is_failure": bool(dimensions),
        "turn_index": turn_index,
        "scenario": scenario,
        "dimensions": dimensions,
    }


def call_model(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, str | None, float, dict[str, Any]]:
    """Call the endpoint once and return content, reasoning, latency, and usage."""

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with Timer() as timer:
        with urllib.request.urlopen(request, timeout=120) as response:
            resp = json.loads(response.read().decode("utf-8"))
    content, reasoning = message_to_payload(resp)
    return content, reasoning, timer.elapsed_ms, usage_to_dict(resp)


def run_demo(args: argparse.Namespace) -> int:
    repo_root = SCRIPT_DIR.parents[1]
    if args.start_serve and not health_ok(args.endpoint):
        start_serve(repo_root)
    if not health_ok(args.endpoint):
        print(
            "M8 endpoint is not healthy. Start it with `bash scripts/serving/serve.sh` "
            f"or rerun this script with --start-serve. Checked {endpoint_readiness_url(args.endpoint)}.",
            file=sys.stderr,
        )
        return 3

    run_id = args.run_id or new_run_id()
    sink = make_trace_sink(
        out_dir=Path(args.out_dir),
        run_id=run_id,
        project=PROJECT,
        provider=args.provider,
    )
    rule_cfg = RuleConfig.from_dict(None)

    turns: list[TraceTurn] = []
    for conv in DEMO_CONVERSATIONS:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for idx, user_text in enumerate(conv["user_turns"], start=1):
            messages.append({"role": "user", "content": user_text})
            content, reasoning, latency_ms, usage = call_model(
                endpoint=args.endpoint,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            scored_content = strip_reasoning(content)
            flags, n_tokens = apply_rules(scored_content, conv["scenario"], rule_cfg)
            failure = classify_failure(flags, conv["scenario"], idx)
            turn = TraceTurn(
                run_id=run_id,
                conversation_id=conv["conversation_id"],
                turn_index=idx,
                scenario=conv["scenario"],
                model=args.model,
                endpoint=args.endpoint,
                prompt_messages=[dict(m) for m in messages],
                content=content,
                scored_content=scored_content,
                reasoning_content=reasoning,
                latency_ms=latency_ms,
                rule_flags=flags,
                n_tokens=n_tokens,
                usage=usage,
                failure=failure,
            )
            sink.record_turn(turn)
            turns.append(turn)
            messages.append({"role": "assistant", "content": scored_content})

    failures = [turn for turn in turns if turn.failure.get("is_failure")]
    summary = {
        "generated_at": turns[-1].timestamp if turns else None,
        "endpoint": args.endpoint,
        "model": args.model,
        "provider": args.provider,
        "n_conversations": len(DEMO_CONVERSATIONS),
        "n_turns": len(turns),
        "n_failure_turns": len(failures),
        "failure_turns": [
            {
                "conversation_id": turn.conversation_id,
                "turn_index": turn.turn_index,
                "scenario": turn.scenario,
                "rule_flags": turn.rule_flags,
                "failure": turn.failure,
                "latency_ms": round(turn.latency_ms, 1),
            }
            for turn in failures
        ],
    }
    artifacts = sink.finalize(summary)
    print(json.dumps({"run_id": run_id, "summary": summary, "artifacts": artifacts}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--provider", choices=["auto", "local", "langfuse", "disabled"], default="auto")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument(
        "--start-serve",
        action="store_true",
        help="Run bash scripts/serving/serve.sh if the M8 endpoint is not already healthy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_demo(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
