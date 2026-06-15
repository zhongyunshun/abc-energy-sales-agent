# M8 concurrency demo (continuous batching)

- endpoint: `http://127.0.0.1:8000/v1` | model: `sales-agent-awq`
- requests: 4 concurrent (streaming), max_tokens=256
- generated_at: 2026-06-15T08:50:57

## Batching efficiency

- warm single-request latency (baseline): **0.4092s** (~114.86 tok/s single stream)
- serial estimate (4x single): **1.6368s**
- concurrent wall-clock (4 streams): **0.6729s**
- **latency speedup: 2.432x**  (4x single / concurrent wall)
- aggregate output throughput: **277.89 tok/s**
- **throughput gain vs single stream: 2.42x**  (robust continuous-batching evidence)

## Latency summary (per-request rollup)

| metric | mean | p50 | p95 |
|---|---:|---:|---:|
| TTFT (s) | 0.1885 | 0.2251 | 0.2300 |
| total (s) | 0.6277 | 0.6259 | 0.6561 |

## Per-request

| id | TTFT (s) | total (s) | ITL mean (s) | tokens |
|---|---:|---:|---:|---:|
| dlg-d0ee1a595d86 | 0.2217 | 0.6277 | 0.0097 | 47 |
| dlg-b81763953222 | 0.2302 | 0.5978 | 0.0097 | 43 |
| dlg-15adb2190982 | 0.2286 | 0.6611 | 0.0096 | 50 |
| dlg-0829a2f26f85 | 0.0736 | 0.6241 | 0.0122 | 47 |
