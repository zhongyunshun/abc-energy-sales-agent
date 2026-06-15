# Data Directory

All files in this directory are **committed build artifacts**: they are small
(megabytes) and keeping them in the repo lets pipeline stages run on a fresh
server checkout without re-downloading sources. Every dataset here can also be
regenerated from configs and scripts; each stage writes a `manifest.json`
(input hashes, config snapshot, git commit, timestamp) next to its outputs so
results are reproducible either way.

## Layout

```
data/
├── raw/          # As-downloaded source files (optional; HF downloads use the HF cache)
├── interim/      # M1 normalized data + M2 synthetic data (pipeline intermediates)
│   ├── normalized.jsonl        # M1 output: cleaned DialogueRecord JSONL
│   ├── normalize_report.json   # M1 per-source input/output and drop-reason counts
│   └── manifest.json           # M1 provenance manifest
└── processed/    # M3 output: train/val/test splits + split_report.json
```

## Data contract

Every dialogue file is JSONL (UTF-8, one object per line) of `DialogueRecord`
objects, defined in `src/sales_agent/common/schema.py`:

```json
{
  "id": "dlg-<12-hex content hash>",
  "source": "hf:goendalf666/sales-conversations",
  "scenario": "objection_handling | info_gathering | cold_open | closing | general",
  "lang": "en",
  "n_turns": 1,
  "meta": {"raw_format": "prefixed_pairs"},
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

`messages` holds at most one system message (first position), then strictly
alternating user/assistant turns ending with assistant. Every record in
`normalized.jsonl` has passed `validate_dialogue()`.

## Collection

### Public dataset (M1 input)

- **Source**: [`goendalf666/sales-conversations`](https://huggingface.co/datasets/goendalf666/sales-conversations)
  (3,412 rows), downloaded at runtime via `datasets.load_dataset` — nothing
  is committed to the repo.
- **What it is**: sales conversations synthesized with GPT-3.5-turbo from a
  sales textbook dataset ("textbooks are all you need" approach). Rows are
  20 string columns (`"0"`..`"19"`) holding `Customer: ...` / `Salesman: ...`
  prefixed turns, with `None` padding.
- **Known quirk**: adjacent customer/salesman exchanges within one row are
  usually *unrelated in topic* (the rows are themed bundles, not coherent
  conversations). We therefore split each row into independent
  **single-turn** dialogues (`format: prefixed_pairs` adapter) instead of
  treating a row as one multi-turn conversation, which would teach the model
  to ignore context. Coherent multi-turn dialogues come from the synthesis
  stage (M2) instead.
- **License**: the dataset declares **no license** on the Hugging Face hub.
  The content itself is GPT-3.5-generated synthetic text with no PII. We use
  it as base training material for this R&D challenge and document that
  status here.

### Synthetic data (M2, planned)

Multi-turn energy-sales dialogues and DPO preference pairs generated via
OpenRouter land in `interim/synthetic_dialogues.jsonl` and
`interim/preference_pairs.jsonl`. See `configs/synthesize.yaml`.

## Processing (M1 normalize)

Reproduce with:

```bash
uv run python scripts/data/normalize.py --config configs/normalize.yaml --smoke  # quick check
uv run python scripts/data/normalize.py --config configs/normalize.yaml          # full run
```

Pipeline (core logic in `src/sales_agent/data/normalize.py`, order is fixed):

1. **Load** each source declared in `configs/normalize.yaml` (local
   json/jsonl or HF dataset; `prefixed_pairs` rows are exploded into
   customer/salesman pairs first).
2. **Convert** to `DialogueRecord`: role mapping (`human/gpt` →
   `user/assistant`), consecutive same-role merge, system prompt hoisted to
   the front. Malformed inputs are dropped as `conversion_failed`.
3. **Clean** (fixed order):
   1. drop empty/too-short turns (< `min_content_chars`), re-merging
      neighbours to preserve alternation;
   2. drop non-English dialogues (langdetect, seeded, system prompt
      excluded) — `non_english`;
   3. replace PII with placeholders: emails → `[EMAIL]`, card numbers →
      `[CARD]`, phone numbers → `[PHONE]`;
   4. truncate trailing non-assistant turns.
4. **Dedup** exactly on a normalized rendering (lowercased, whitespace
   collapsed, role-prefixed) — first occurrence wins; the same hash is the
   record `id` — `duplicate`.
5. **Tag scenario** with keyword rules from the config; no hit falls back to
   `general` (accurate labels arrive with M2 synthetic data).
6. **Validate** every surviving record with `validate_dialogue()`; failures
   are dropped as `validation_failed` and counted.

All drop reasons are counted per source in `normalize_report.json`.

### Current full-run snapshot (2026-06-12)

3,412 rows → 20,927 exploded pairs → **20,732 records** written.
Dropped: 192 `duplicate`, 3 `non_english`; 2 PII replacements.
Scenario distribution: 864 objection_handling, 318 info_gathering,
37 cold_open, 26 closing, 19,487 general.
