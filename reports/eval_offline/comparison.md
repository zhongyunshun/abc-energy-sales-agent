# M9 offline-eval — three-group rule-metric comparison

Same engine (vLLM, compressed-tensors W4A16 AWQ on the RTX 4070), same generation
params (temperature=0, max_tokens=256), same 650-sample test batch (M3
`data/processed/test.jsonl`). Length is a deterministic word/punct token count
(`rules.count_tokens`); over_length budget = 120 tokens (voice replies must be
short, the M9 offline-eval target). Numbers are real (not curated).

- samples per group: 650
- groups: **base** = `unsloth/Qwen3-4B-Instruct-2507` (untuned instruct),
  **sft** = base + SFT LoRA (merged), **dpo** = base + SFT→DPO (merged, the
  deployed model)

## Overall rule trigger rates (lower is better)

| rule | base | sft | dpo |
|---|---:|---:|---:|
| made_up_price | 0.1015 | 0.0015 | 0.0000 |
| over_length | 0.6785 | 0.0000 | 0.0000 |
| role_break | 0.0892 | 0.0000 | 0.0000 |
| no_question_in_gathering | 0.0815 | 0.0646 | 0.0615 |

## Reply length (tokens)

| stat | base | sft | dpo |
|---|---:|---:|---:|
| mean | 186.28 | 38.36 | 37.52 |
| p50 | 214.5 | 37.0 | 36.0 |
| p90 | 276.0 | 54.0 | 53.0 |
| p95 | 280.0 | 59.0 | 56.0 |
| max | 311 | 117 | 111 |

## no_question_in_gathering within the info_gathering scenario

| group | n (info_gathering) | rate |
|---|---:|---:|
| base | 82 | 0.6463 |
| sft | 82 | 0.5122 |
| dpo | 82 | 0.4878 |

## Observations

- **SFT does the heavy lifting**: made_up_price 10.15%→0.15%, over_length
  67.85%→0%, role_break 8.92%→0%, median length 214.5→37 tokens. The untuned base
  is verbose and hallucinates concrete prices; SFT makes replies short,
  in-persona, and price-clean.
- **DPO nudges further but mildly**: made_up_price 0.15%→0% (the one residual SFT
  price violation removed), with a small length / question-asking improvement. The
  SFT→DPO behavioral delta is small here (SFT is already well-aligned and greedy
  head-room is limited) — consistent with the M5 DPO behavior-probe finding (risk
  board 2026-06-13).
- **no_question_in_gathering** is the only non-trivial residual for sft/dpo:
  ~49–51% of info_gathering replies contain no literal "?" (the rule is strict on
  a question mark). DPO is marginally better than SFT (0.4878 vs 0.5122). This is a
  behavior signal for M10/judge, not a price/length/role failure.

## Reproduce

```
bash scripts/serving/serve.sh                                  # serve the AWQ model
uv run python scripts/eval/run_offline_eval.py \
    --config configs/eval_offline.yaml --model-tag <base|sft|dpo>
```

Per-group products: `reports/eval_offline/<tag>/{results.jsonl,summary.json,manifest.json}`.
