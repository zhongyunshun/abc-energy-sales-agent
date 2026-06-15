# M9 Offline Evaluation — Task List

> English snapshot of `doc/tasks/m09-eval-offline.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: M0, test.jsonl (or fixtures); the endpoint path needs M8

## Tasks

- [x] T9.1 Implement `build_eval_samples()`: take the context before the last assistant turn as the prompt and the last turn as gold + unit tests (multi-turn / single-turn / with-system boundaries)
- [x] T9.2 Implement the four rule metrics in `evals/rules.py` (made_up_price / over_length / role_break / no_question_in_gathering), with full positive/negative fixture unit-test coverage per rule
- [x] T9.3 Author `configs/eval_offline.yaml` + a thin CLI: async batch generation against an OpenAI-compatible endpoint (fixed temperature=0) → rule tagging → `results.jsonl` + `summary.json` (per-scenario trigger rates, length percentiles); endpoint-client mock unit tests
- [x] T9.4 Implement a local transformers inference fallback path (`--local-adapter`, supports evaluating base and not-yet-deployed adapters; optional PPL computation)
- [x] T9.5 Run smoke (10 rows) against the real M8 service; then the full three groups: base / SFT / SFT+DPO each produce a results + summary set

## Definition of Done (DoD)

Rule and sample-construction unit tests green; the three `summary.json` are produced with consistent methodology (same evaluation sample batch, same generation params).
