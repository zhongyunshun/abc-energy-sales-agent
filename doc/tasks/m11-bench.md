# M11 Load Test (Locust) — Task List

> English snapshot of `doc/tasks/m11-bench.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Design refs: `detailed-design.md` §3-M11 | Prerequisites: M0, M8 service online; analyze.py already done in T8.3

## Tasks

- [x] T11.1 Author `locustfile.py`: stream-call `/v1/chat/completions`, a length-bucketed round-robin prompt pool, per-chunk timing reporting of custom metrics (ttft_ms / itl_ms / output_tokens); the timing logic is extracted into a unit-tested function
- [x] T11.2 Author the `run_bench.py` orchestrator: concurrency tiers [1,4,8,16,32] run headless per tier (120s/tier + 15s warmup discarded), collect raw CSV
- [x] T11.3 Implement aggregation and plotting: `bench_summary.csv` (§2.4 contract columns) + three PNGs (throughput vs concurrency, TTFT p50/p95, ITL); aggregation-value unit tests (reuse analyze.py)
- [x] T11.4 Run a single-tier 30s smoke against the real service first, then the full ladder load test; confirm the relationship between the 32-concurrency tier's error rate and the max_num_seqs config, and re-tune M8 params and retest if needed

## Definition of Done (DoD)

`bench_summary.csv` + three plots produced (core performance-report data); no OOM/crash recorded during the load test.
