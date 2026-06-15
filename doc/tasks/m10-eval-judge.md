# M10 LLM-as-a-Judge — Task List

> English snapshot of `doc/tasks/m10-eval-judge.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Design refs: `detailed-design.md` §3-M10 | Prerequisites: M0, M9 three-group results (or fixtures)

## Tasks

- [x] T10.1 Author the Referee prompt `configs/prompts/judge.j2`: a 4-dimension 1–5 rubric (coherence / sales_logic / professionalism / hallucination-free) + per-score anchors + few-shot calibration samples + forced JSON + anti-length/politeness-bias instructions
- [x] T10.2 Implement `select_judge_samples()`: take the same id batch across the three groups, a scenario-stratified sample of 100, fixed seed + unit tests (same-id-set assertion)
- [x] T10.3 Implement `parse_judge_response()` and `aggregate_scores()`: parsing (valid / missing-field / out-of-range / non-JSON) + model_tag × scenario cross aggregation (mean gap <0.3 labeled "no significant difference") + unit tests
- [x] T10.4 Author `configs/eval_judge.yaml` + a thin CLI: call the judge model via OpenRouter (default a different source from the synthesis model), ≤2 retries on parse failure, `--smoke` 5 per group; mock unit tests with zero real API
- [x] T10.5 Real run of the three-group comparison → `scores.jsonl` + `comparison.md` (base vs SFT vs SFT+DPO four-dimension comparison table, core report material)

## Definition of Done (DoD)

Unit tests green; `comparison.md` produced and the three groups' sample ids are identical; judge cost recorded.
