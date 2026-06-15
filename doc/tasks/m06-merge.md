# M6 Adapter Merge — Task List

> English snapshot of `doc/tasks/m06-merge.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: M0, M5 (or M4) adapter

## Tasks

- [x] T6.1 Implement the consistency comparator (compare PEFT-inference vs merged-inference greedy outputs line by line, supporting exact / first-64-token modes) + unit tests (the comparison logic itself)
- [x] T6.2 Author `configs/merge.yaml` + `scripts/training/merge_adapter.py`: BF16-load base → `merge_and_unload` → save safetensors + tokenizer + chat template + manifest; config includes a `device_map=cpu` fallback path
- [x] T6.3 Implement an 8-fixed-prompt merge-consistency smoke check: exit code 2 and print the diff on mismatch
- [x] T6.4 Run on the real DPO adapter in the container: BF16 merge succeeds under 12GB (fall back to CPU on failure), the consistency check passes

## Definition of Done (DoD)

`models/merged/` produced and directly loadable by transformers for generation; the consistency-check pass is recorded in the manifest.
