# M3 Split & Leakage Prevention — Task List

> English snapshot of `doc/tasks/m03-split.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: M0 (inputs may use fixtures in place of M1/M2 products)

## Tasks

- [x] T3.0 Implement `downsample_m1()` (M1 downsampling, after global dedup and before split): the always-kept high-value union = (scenario ∈ {objection_handling, info_gathering}) ∪ (energy-keyword `\b` word-boundary hit); randomly fill from the remaining `general` with a fixed seed up to `target = round(ratio × actual M2 count)`; when the high-value set ≥ target, keep all, flag it in the report, and pause for confirmation. The energy-keyword list and matching method go into `configs/split.yaml` + unit tests (high-value always kept, different ratio values, high-value-exceeds-target boundary, same-seed idempotency)
- [x] T3.1 Implement `minhash_dedup()`: 5-gram shingle → MinHash(128) → LSH clustering (threshold 0.85) → in-cluster retention policy (prefer real data, longer dialogues) + unit tests (construct a known near-duplicate fixture to verify recall and retention priority)
- [x] T3.2 Implement `stratified_split()`: stratify key `(scenario, turn_bucket)`, merge small strata (<20), split by complete dialogue 90/5/5 + unit tests (ratios, distribution, same-seed idempotency)
- [x] T3.3 Implement `assert_no_leakage()` cross-split MinHash check + unit tests (injected cross-split duplicates must be detected)
- [x] T3.4 Author `configs/split.yaml` + a thin CLI: merge M1+M2 inputs → global dedup → split → leakage assertion (exit code 2 on violation) → write `{train,val,test}.jsonl` + `split_report.json` (§2.3 contract: counts/distribution/leakage_check; distribution deviation >3pp emits WARN)
- [x] T3.5 Run on real data: `cross_split_dups == 0`, consistent distribution across the three sets, report data feeds the downstream report material

## Definition of Done (DoD)

Unit tests green; real data produces train/val/test and the leakage assertion passes; split_report.json conforms to the contract.
