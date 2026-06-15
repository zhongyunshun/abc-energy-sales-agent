# M8 concurrency demo (continuous batching)

- endpoint: `http://127.0.0.1:8000/v1` | model: `sales-agent-awq`
- requests: 16 concurrent (streaming), max_tokens=256
- generated_at: 2026-06-14T10:31:32

## Batching efficiency

- warm single-request latency (baseline): **0.4678s** (~100.48 tok/s single stream)
- serial estimate (16x single): **7.4841s**
- concurrent wall-clock (16 streams): **0.739s**
- **latency speedup: 10.128x**  (16x single / concurrent wall)
- aggregate output throughput: **917.48 tok/s**
- **throughput gain vs single stream: 9.13x**  (robust continuous-batching evidence)

## Latency summary (per-request rollup)

| metric | mean | p50 | p95 |
|---|---:|---:|---:|
| TTFT (s) | 0.1014 | 0.1057 | 0.1151 |
| total (s) | 0.5886 | 0.5852 | 0.6674 |

## Per-request

| id | TTFT (s) | total (s) | ITL mean (s) | tokens |
|---|---:|---:|---:|---:|
| dlg-d0ee1a595d86 | 0.1193 | 0.6504 | 0.0126 | 47 |
| dlg-b81763953222 | 0.1041 | 0.5972 | 0.0130 | 43 |
| dlg-15adb2190982 | 0.1031 | 0.6176 | 0.0129 | 45 |
| dlg-0829a2f26f85 | 0.0775 | 0.6466 | 0.0126 | 47 |
| dlg-90392f1a0e2e | 0.1136 | 0.5689 | 0.0130 | 40 |
| dlg-e42b1638d602 | 0.1133 | 0.5927 | 0.0130 | 42 |
| dlg-767d9ddb1c2e | 0.1117 | 0.5668 | 0.0130 | 40 |
| dlg-0ac2fc43a55c | 0.1103 | 0.5144 | 0.0130 | 36 |
| dlg-6e418931a42a | 0.1094 | 0.5777 | 0.0130 | 41 |
| dlg-af968210a24e | 0.1079 | 0.7183 | 0.0122 | 55 |
| dlg-64aa3c6e2f09 | 0.1072 | 0.5491 | 0.0130 | 39 |
| dlg-bff3c8261c7f | 0.0697 | 0.5366 | 0.0130 | 38 |
| dlg-b763e1483b5c | 0.0674 | 0.4825 | 0.0130 | 34 |
| dlg-94dd620217c8 | 0.1041 | 0.5950 | 0.0129 | 43 |
| dlg-f3af9e2977a8 | 0.1022 | 0.5708 | 0.0130 | 41 |
| dlg-42e17cf2f6ac | 0.1022 | 0.6330 | 0.0126 | 47 |
