# M2 Data Synthesis — Task List

> English snapshot of `doc/tasks/m02-synthesize.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: M0 (M1 not required; the contracts are independent)

## Tasks

- [x] T2.1 Author the prompt templates `configs/prompts/synth_dialogue.j2` (force JSON `messages` output; the salesperson must not invent prices) and `synth_preference.j2` (chosen/rejected pair from the same context)
- [x] T2.2 Implement `expand_task_matrix()`: Cartesian product scenario × persona × objection_type × outcome + quota sampling + unit tests (count vs quota assertions)
- [x] T2.3 Implement the `parse_and_validate()` generation-side quality gate: JSON parse → schema check → min-turns → role alternation → price-number regex rejection; preference pairs add a chosen/rejected divergence lower bound + unit tests (full coverage of bad-output fixtures)
- [x] T2.4 Implement `run_synthesis()` asyncio concurrency orchestration: semaphore throttling, ≤2 retries on failure, cumulative token cost + fault-injection unit tests via `fake_openrouter` (retries, abandon counts)
- [x] T2.5 Author `configs/synthesize.yaml` + a thin CLI (`--mode dialogues|preferences --smoke`); `--smoke` runs 2 per scenario against the real API
- [x] T2.6 Full generation: 1000–2000 dialogues → `synthetic_dialogues.jsonl`; ~300 preference pairs → `preference_pairs.jsonl`; record the cost report; manually spot-check 10 for quality

## Definition of Done (DoD)

Unit tests green (zero real API); smoke and full products all pass contract validation; cost within budget and recorded.
