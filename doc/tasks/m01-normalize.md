# M1 Data Normalization — Task List

> English snapshot of `doc/tasks/m01-normalize.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Design refs: `detailed-design.md` §3-M1 | Prerequisites: M0

## Tasks

- [x] T1.1 Implement `data/normalize.py::alpaca_to_dialogue()` + unit tests (valid sample, missing fields, empty output)
- [x] T1.2 Implement `sharegpt_to_dialogue()`: role mapping, consecutive same-role merge, system hoisted to first + unit tests (out-of-order roles, consecutive gpt turns)
- [x] T1.3 Implement the `clean_dialogue()` cleaning chain: drop empty/too-short turns → drop non-English → PII regex replacement (phone/email/card) → truncate trailing non-assistant turns + unit tests (positive/negative per rule)
- [x] T1.4 Implement exact dedup (normalized-text hash) and `tag_scenario()` keyword tagging + unit tests (idempotency; fall back to `general` when undecidable)
- [x] T1.5 Author `configs/normalize.yaml` + a thin CLI `scripts/data/normalize.py`: multi-source load (local / HF datasets) → convert → clean → write `data/interim/normalized.jsonl` + `normalize_report.json` (per-source in/out counts, drop-reason counts)
- [x] T1.6 Validate candidate public datasets (`goendalf666/sales-conversations`, etc.): downloadable, license usable, CLI runs and produces a real normalized.jsonl; if unusable, remove it from config and record it in progress (switch to the full-synthesis route)

## Definition of Done (DoD)

Unit tests green; the real-data CLI runs and all output passes `DialogueRecord` validation; per-drop-reason counts in the report are reasonable.
