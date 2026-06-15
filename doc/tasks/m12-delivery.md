# M12 Delivery Wrap-up — Task List

> English snapshot of `doc/tasks/m12-delivery.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Design refs: `proposal.md` §5 deliverables, §8 acceptance criteria; `detailed-design.md` §4.1, §5.3 | Prerequisites: M1–M11 main body complete

## Tasks

- [ ] T12.1 Author `scripts/run_e2e_smoke.sh`: a serial M1→M11 full-chain `--smoke`, one-command pipeline-connectivity verification in a clean environment
- [x] T12.2 Author the English `README.md`: tech-selection rationale (model / LoRA hyperparams / AWQ reasoning), per-stage uv and Docker run instructions, quantization trade-off analysis (citing M7/M11 data), resource-isolation note (design §3-C3)
- [x] T12.3 Consolidate the performance report: per the design §5.3 report map, gather loss curves, DPO behavior comparison, size comparison, load-test charts, rule metrics and judge-score comparison, integrated into the README or `reports/REPORT.md`
- [ ] T12.4 Author the English whitepaper `doc/whitepaper.md`: Bonus 1 thousand-concurrency blueprint (multi-replica vLLM / gateway / KV cache / PagedAttention / Speculative Decoding / SLO), Bonus 2 NVFP4 route and quantization Smoke Test design, Bonus 3b Langfuse tracing plan, production-grade resource isolation (MIG/MPS/K8s)
- [x] T12.5 Acceptance cross-check: verify proposal §8's five acceptance criteria one by one (§8-4 whitepaper coverage deferred to bonus closeout), tidy git history, repo public-readiness check (no key leaks, large files ignored)

## Definition of Done (DoD)

A clean environment reproduces the full chain per the README; all proposal §8 acceptance criteria are checked and passing.
