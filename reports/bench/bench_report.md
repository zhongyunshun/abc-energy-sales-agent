# M11 load test (Locust, AWQ/INT4)

- endpoint: `http://127.0.0.1:8000` | served INT4 (compressed-tensors) on RTX 4070, M8 service
- generated_at: 2026-06-14T22:20:18

## Steady-state metrics vs concurrency

| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | tok/s | req/s | error % | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45.6 | 90.1 | 9.2 | 9.7 | 102.1 | 2.50 | 0.0 | 263 |
| 4 | 55.2 | 112.5 | 11.2 | 12.3 | 334.2 | 8.00 | 0.0 | 840 |
| 8 | 68.2 | 122.2 | 13.0 | 15.0 | 575.9 | 13.78 | 0.0 | 1447 |
| 16 | 112.6 | 177.7 | 18.3 | 21.2 | 820.2 | 19.72 | 0.0 | 2071 |
| 32 | 867.7 | 1034.3 | 18.2 | 21.3 | 835.3 | 20.09 | 0.0 | 2109 |

## Plots

- ![throughput](throughput_vs_concurrency.png)
- ![ttft](ttft_vs_concurrency.png)
- ![itl](itl_vs_concurrency.png)

## Notes

- Concurrency is closed-loop (N users, no think-time) so in-flight load ~= N. 32 exceeds the M8 cap max_num_seqs=16 on purpose: vLLM queues past 16, so the 16->32 step is where TTFT rises and throughput plateaus (the real knee).
- FP16 vs INT4 TTFT/ITL comparison NOT measured: the merged FP16 model (8.045GB) does not fit alongside the display on the 4070 (~7.9GB free), so only the INT4 (AWQ/compressed-tensors) service is benchmarked here. Size/theory comparison: M7 manifest (8.045GB FP16 -> 2.666GB INT4, 3.02x); quality: M7's 5 FP16-vs-INT4 probes (no visible regression). Handed to M12 README.
