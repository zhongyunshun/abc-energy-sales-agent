# M10 LLM-as-a-Judge — three-group comparison

Blind scoring of base vs SFT vs SFT+DPO replies (M9 `results.jsonl`, the same 100 ids per group, scenario-stratified, seed=42). Each judge sees only the dialogue + candidate reply, never the model tag. Four dimensions, 1–5 (5 best); `hallucination` is hallucination-free (5 = invents no price/fact). Cells are mean±std.

- judges (non-Google, cross-validation): anthropic/claude-sonnet-4.6, openai/gpt-5.4
- samples scored per group: 100
- scores collected: 600 / 600 attempted
- no-significant-difference threshold: mean gap < 0.3

## Judge: `anthropic/claude-sonnet-4.6` (n=300)

### Overall (mean±std, higher is better)

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 2.95±1.66 | 3.44±1.40 | 3.43±1.43 |
| sales_logic | 2.27±1.47 | 2.69±1.33 | 2.64±1.31 |
| professionalism | 2.48±1.49 | 3.40±1.49 | 3.38±1.54 |
| hallucination | 4.11±1.09 | 4.28±1.06 | 4.22±1.19 |

Pairwise overall-mean differences:
  - coherence: base vs sft Δ=-0.49
  - coherence: base vs dpo Δ=-0.48
  - coherence: sft vs dpo Δ=+0.01 → no significant difference
  - sales_logic: base vs sft Δ=-0.42
  - sales_logic: base vs dpo Δ=-0.37
  - sales_logic: sft vs dpo Δ=+0.05 → no significant difference
  - professionalism: base vs sft Δ=-0.92
  - professionalism: base vs dpo Δ=-0.90
  - professionalism: sft vs dpo Δ=+0.02 → no significant difference
  - hallucination: base vs sft Δ=-0.17 → no significant difference
  - hallucination: base vs dpo Δ=-0.11 → no significant difference
  - hallucination: sft vs dpo Δ=+0.06 → no significant difference

### By scenario (mean per dimension)

**closing**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 5.00±0.00 | 5.00±0.00 | 5.00±0.00 |
| sales_logic | 4.25±0.43 | 4.00±0.71 | 4.50±0.50 |
| professionalism | 4.25±0.83 | 5.00±0.00 | 5.00±0.00 |
| hallucination | 4.75±0.43 | 5.00±0.00 | 5.00±0.00 |

**cold_open**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 5.00±0.00 | 4.80±0.40 | 4.80±0.40 |
| sales_logic | 4.60±0.49 | 4.40±0.49 | 4.00±0.63 |
| professionalism | 4.40±0.49 | 5.00±0.00 | 5.00±0.00 |
| hallucination | 4.20±1.60 | 4.80±0.40 | 5.00±0.00 |

**general**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 1.98±1.16 | 2.72±1.19 | 2.70±1.22 |
| sales_logic | 1.35±0.79 | 1.97±0.89 | 1.95±0.94 |
| professionalism | 1.60±0.86 | 2.60±1.20 | 2.57±1.27 |
| hallucination | 3.87±1.01 | 3.92±1.17 | 3.78±1.32 |

**info_gathering**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 4.54±1.08 | 4.62±0.84 | 4.62±0.84 |
| sales_logic | 3.38±0.92 | 3.54±0.93 | 3.46±1.08 |
| professionalism | 3.77±1.05 | 4.69±1.07 | 4.62±1.08 |
| hallucination | 4.23±1.42 | 4.85±0.53 | 4.85±0.53 |

**objection_handling**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 4.00±1.37 | 4.28±1.10 | 4.28±1.15 |
| sales_logic | 3.44±1.30 | 3.72±1.37 | 3.56±1.21 |
| professionalism | 3.56±1.46 | 4.33±1.11 | 4.39±1.16 |
| hallucination | 4.67±0.67 | 4.78±0.53 | 4.83±0.50 |

## Judge: `openai/gpt-5.4` (n=300)

### Overall (mean±std, higher is better)

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 3.04±1.48 | 3.65±1.32 | 3.69±1.36 |
| sales_logic | 2.19±1.26 | 2.68±1.18 | 2.74±1.22 |
| professionalism | 2.82±1.37 | 3.80±1.36 | 3.80±1.39 |
| hallucination | 4.50±1.06 | 4.34±0.84 | 4.36±0.90 |

Pairwise overall-mean differences:
  - coherence: base vs sft Δ=-0.61
  - coherence: base vs dpo Δ=-0.65
  - coherence: sft vs dpo Δ=-0.04 → no significant difference
  - sales_logic: base vs sft Δ=-0.49
  - sales_logic: base vs dpo Δ=-0.55
  - sales_logic: sft vs dpo Δ=-0.06 → no significant difference
  - professionalism: base vs sft Δ=-0.98
  - professionalism: base vs dpo Δ=-0.98
  - professionalism: sft vs dpo Δ=+0.00 → no significant difference
  - hallucination: base vs sft Δ=+0.16 → no significant difference
  - hallucination: base vs dpo Δ=+0.14 → no significant difference
  - hallucination: sft vs dpo Δ=-0.02 → no significant difference

### By scenario (mean per dimension)

**closing**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 4.25±0.43 | 5.00±0.00 | 5.00±0.00 |
| sales_logic | 4.00±0.00 | 4.50±0.50 | 4.50±0.50 |
| professionalism | 4.25±0.83 | 5.00±0.00 | 5.00±0.00 |
| hallucination | 4.75±0.43 | 5.00±0.00 | 4.75±0.43 |

**cold_open**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 4.80±0.40 | 4.80±0.40 | 5.00±0.00 |
| sales_logic | 4.00±0.00 | 4.20±0.40 | 4.20±0.40 |
| professionalism | 4.40±0.49 | 5.00±0.00 | 5.00±0.00 |
| hallucination | 4.20±1.60 | 4.80±0.40 | 5.00±0.00 |

**general**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 2.25±1.15 | 3.05±1.23 | 3.10±1.30 |
| sales_logic | 1.40±0.73 | 2.07±0.89 | 2.12±0.95 |
| professionalism | 2.05±0.99 | 3.17±1.28 | 3.23±1.36 |
| hallucination | 4.53±1.01 | 4.02±0.87 | 4.00±0.97 |

**info_gathering**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 4.46±0.84 | 4.69±0.46 | 4.85±0.36 |
| sales_logic | 3.31±0.82 | 3.46±0.75 | 3.69±0.72 |
| professionalism | 4.00±0.78 | 4.77±0.80 | 4.77±0.80 |
| hallucination | 4.23±1.42 | 4.92±0.27 | 4.92±0.27 |

**objection_handling**

| dimension | base | sft | dpo |
|---|---|---|---|
| coherence | 3.89±1.29 | 4.28±1.15 | 4.17±1.17 |
| sales_logic | 3.11±1.05 | 3.33±1.05 | 3.33±1.11 |
| professionalism | 3.78±1.18 | 4.61±0.95 | 4.39±1.16 |
| hallucination | 4.61±0.76 | 4.72±0.65 | 4.89±0.46 |

## Judge cost

| judge model | requests | prompt tok | completion tok | est. USD |
|---|---:|---:|---:|---:|
| anthropic/claude-sonnet-4.6 | 300 | 503882 | 58371 | $2.3872 |
| openai/gpt-5.4 | 300 | 444372 | 46653 | $1.8107 |
| **total** | | | | **$4.1979** |
