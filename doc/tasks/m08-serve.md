# M8 vLLM Serving — Task List

> English snapshot of `doc/tasks/m08-serve.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: M0, M7 AWQ product

## Tasks

- [x] T8.1 Author the `compose.yaml` serve service: official `vllm/vllm-openai` pinned-version image, AWQ model volume mount, 12GB tuning params (gpu_mem_util 0.90, max_model_len 4096, max_num_seqs 32, prefix caching)
- [x] T8.2 Author `scripts/serving/serve.sh`: compose startup + `/health` polling health check (120s timeout, exit code 3)
- [x] T8.3 Implement `bench/analyze.py` streaming-timing aggregation pure functions (compute TTFT/ITL/total latency from timestamped chunk streams) + unit tests (reused by M11)
- [x] T8.4 Author `concurrency_demo.py`: asyncio 16 concurrent streaming requests (prompts from the test set) → print a per-request TTFT/total-latency summary table
- [x] T8.5 On-hardware verification: the service starts successfully, the demo runs, 16-way total time << 16× single-stream (evidence that continuous batching works; screenshot/data stored under reports/)

## Definition of Done (DoD)

`serve.sh` brings up the service in one command; the demo outputs concurrency-efficiency evidence; analyze.py unit tests green.
