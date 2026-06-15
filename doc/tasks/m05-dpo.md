# M5 DPO Training — Task List

> English snapshot of `doc/tasks/m05-dpo.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: M0, M4 adapter, M2 preference pairs (or fixtures)

## Tasks

- [x] T5.1 Implement `formatting.py` preference-pair conversion: `PreferencePair` → DPO format (prompt/chosen/rejected text) + unit tests (incl. an error when the context does not end on a user turn)
- [x] T5.2 Author `configs/dpo.yaml` (beta 0.1, lr 5e-6, 1 epoch) + `scripts/training/train_dpo.py`: load base+SFT adapter, reference model via a PEFT shared-weights scheme (no extra full-model VRAM)
- [x] T5.3 Freeze 20 probe prompts as a fixture (prone to eliciting pushy / rate hallucination), implement pre/post greedy comparison output `reports/training/dpo_behavior_diff.md`
- [x] T5.4 Container `--smoke` run; confirm peak VRAM within budget
- [x] T5.5 Full training: loss/margins curves produced; behavior_diff should at least show visible regression of pushy/hallucination behavior (subjective judgment, record the conclusion)

## Definition of Done (DoD)

Conversion unit tests green; DPO adapter produced and loadable; `dpo_behavior_diff.md` and the margins curve feed the report material.
