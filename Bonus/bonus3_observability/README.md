# Bonus 3 Observability

Bonus 3 has two parts:

- **3a load testing:** already implemented by M11 Locust. This directory only
  cites the existing M11 artifacts; it does not re-run or rewrite the load test.
- **3b reasoning trace observability:** implemented here as a minimal Langfuse
  integration with local JSONL/HTML fallback for multi-turn sales conversations.

## What Was Implemented

Files in this directory:

- `trace_sink.py`: trace sink abstraction. It always writes local JSONL and HTML
  exports. If Langfuse credentials and SDK are available, it also reports a trace
  and one generation per assistant turn.
- `run_multiturn_trace_demo.py`: runs three multi-turn sales conversations against
  the M8 OpenAI-compatible vLLM endpoint and records prompt, content,
  `reasoning_content`, latency, token usage, rule flags, and failure annotations.
- `eval_trace_hook.py`: optional M9-style hook. It is disabled by default and can
  be imported by a future eval runner without changing existing M9 code.

No files outside `Bonus/bonus3_observability/` are changed by this integration.

## Provider Choice

I chose **Langfuse** over W&B because the trace/generation data model fits this
task directly: one trace per multi-turn conversation and one generation per
assistant turn.

This run used the **local fallback** because no Langfuse credentials were
configured in the shell. The local fallback is still the canonical demo artifact:

- JSONL trace:
  `Bonus/bonus3_observability/traces/bonus3-20260615T174732Z-a6fa992d.jsonl`
- Summary:
  `Bonus/bonus3_observability/traces/bonus3-20260615T174732Z-a6fa992d_summary.json`
- HTML export:
  `Bonus/bonus3_observability/exports/bonus3-20260615T174732Z-a6fa992d_trace_report.html`

To report to Langfuse, install the SDK in the host environment and set:

```powershell
$env:LANGFUSE_PUBLIC_KEY = "..."
$env:LANGFUSE_SECRET_KEY = "..."
$env:LANGFUSE_HOST = "https://cloud.langfuse.com"  # or local self-hosted URL
$env:PYTHONPATH = "src"
python Bonus\bonus3_observability\run_multiturn_trace_demo.py --provider langfuse
```

If `--provider auto` is used, Langfuse is attempted only when the two key
environment variables exist; otherwise the script writes local artifacts.

## How To Run

Start the M8 service first:

```powershell
docker compose -f docker/compose.yaml up -d serve
```

Equivalent normal project path:

```bash
bash scripts/serving/serve.sh
```

Then run the demo from the repo root:

```powershell
$env:PYTHONPATH = "src"
python Bonus\bonus3_observability\run_multiturn_trace_demo.py --endpoint http://localhost:8000/v1 --provider auto
```

The script uses standard-library HTTP against `/v1/chat/completions`, so the demo
does not require the OpenAI Python SDK. It records the endpoint's
`reasoning_content` field when present. In the observed M8 run,
`reasoning_content` was `null` for all turns, matching the M8 finding that the
Qwen3 parser strips the learned empty `<think></think>` prefix and leaves clean
assistant content.

## Trace Fields

Each JSONL row records:

- `run_id`, `conversation_id`, `turn_index`, `scenario`
- `prompt_messages`
- model `content`
- `scored_content` after defensive reasoning-prefix stripping
- `reasoning_content`
- `latency_ms`
- endpoint `usage`
- M9-style `rule_flags`
- `failure` annotation

The rule flags reuse the M9 rule semantics:

- `made_up_price`
- `over_length`
- `role_break`
- `no_question_in_gathering`

## Failure Localization Example

Run:

```text
bonus3-20260615T174732Z-a6fa992d
```

Observed totals:

- Conversations: 3
- Assistant turns: 7
- Failure turns: 1

The localized failure is:

- Conversation: `conv-adversarial-no-question`
- Turn: `2`
- Scenario: `info_gathering`
- Failed dimension: `info_gathering`
- Rule: `no_question_in_gathering`
- Latency: `331.6 ms`

Trace evidence:

```json
{
  "rule_flags": {
    "made_up_price": false,
    "over_length": false,
    "role_break": false,
    "no_question_in_gathering": true
  },
  "failure": {
    "is_failure": true,
    "turn_index": 2,
    "scenario": "info_gathering",
    "dimensions": [
      {
        "dimension": "info_gathering",
        "rule": "no_question_in_gathering",
        "detail": "The assistant did not ask a discovery question in an info-gathering turn."
      }
    ]
  }
}
```

The prompt at that turn explicitly pressured the model to avoid question marks
while still being categorized as `info_gathering`. The assistant complied with
the user pressure and produced a proposal-style next step rather than asking a
discovery question:

```text
Thank you for that information. With your usage and decision-making role, I would
like to prepare a detailed, no-obligation proposal that specifically addresses
your cost-saving goals.
```

The trace makes the failure easy to locate: it is not a latency problem, not a
reasoning-parser artifact, and not a price hallucination. It is a multi-turn
instruction-following conflict where the current user instruction overrode the
info-gathering behavior expected by the rule metric.

## Optional M9 Hook

`eval_trace_hook.py` is the reserved Bonus 3b interface for future offline eval
instrumentation. It is deliberately opt-in:

```python
from Bonus.bonus3_observability.eval_trace_hook import (
    build_eval_trace_sink,
    record_eval_generation,
)

sink = build_eval_trace_sink(enabled=False)  # default: no tracing

# Inside a future eval loop, after one generation is available:
record_eval_generation(
    sink,
    sample=sample,
    output=output,
    model="sales-agent-awq",
    endpoint="http://127.0.0.1:8000/v1",
    latency_ms=latency_ms,
)
```

Because the hook is import-only and default-disabled, M9 behavior and outputs do
not change unless a caller explicitly enables tracing.

## Bonus 3a: M11 Load Test Reference

Bonus 3a is already implemented by M11 under `scripts/bench/` and
`reports/bench/`; it is not duplicated here.

Existing M11 artifacts:

- `reports/bench/bench_summary.csv`
- `reports/bench/bench_report.md`
- `reports/bench/raw_1.csv`, `raw_4.csv`, `raw_8.csv`, `raw_16.csv`, `raw_32.csv`
- `reports/bench/throughput_vs_concurrency.png`
- `reports/bench/ttft_vs_concurrency.png`
- `reports/bench/itl_vs_concurrency.png`

M11 measured the AWQ/INT4 M8 endpoint on the RTX 4070 with Locust concurrency
steps `[1, 4, 8, 16, 32]`. The key result is the 16 to 32 knee:

| concurrency | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | throughput tok/s | error rate |
|---:|---:|---:|---:|---:|---:|
| 16 | 112.57 | 177.67 | 18.295 | 820.19 | 0.0 |
| 32 | 867.73 | 1034.34 | 18.204 | 835.31 | 0.0 |

Throughput plateaus from `820.19` to `835.31 tok/s`, while TTFT jumps from
`112.57 ms` to `867.73 ms` because 32 concurrent users exceed M8
`max_num_seqs=16` and vLLM queues requests. This confirms the practical 12GB
serving knee without re-running the benchmark.
