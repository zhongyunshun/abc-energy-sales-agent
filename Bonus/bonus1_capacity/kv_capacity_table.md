# KV-cache capacity -- qwen3 (AWQ INT4) on A100 40GB

## Model structure (read from models/quantized/awq/config.json)

- num_layers = 36
- num_kv_heads = 8 (GQA; attn heads = 32)
- head_dim = 128
- KV dtype = 2 bytes (bf16)
- max_position_embeddings = 262144

## Per-token KV

`bytes/token = num_layers x 2 x num_kv_heads x head_dim x dtype_bytes`
= 36 x 2 x 8 x 128 x 2 = **147,456 B = 144 KiB/token**

## Memory budget on a single A100 40GB

- usable @ util 0.9 = 36.0 GiB
- AWQ INT4 weights = 2.48 GiB  |  bf16 weights = 7.49 GiB
- non-KV overhead = 2.0 GiB
- **KV budget (AWQ) = 31.5 GiB** | KV budget (bf16) = 26.5 GiB

## Worst case: 1000 sessions each holding full context

| ctx | KV/session | KV x 1000 | max sessions (AWQ) | max sessions (bf16) | A100s for 1000 (AWQ) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 288 MiB | 281 GiB | 112 | 94 | 8.9 |
| 4096 | 576 MiB | 562 GiB | 56 | 47 | 17.8 |

## Realistic voice scenario (AWQ weights)

- active fraction = 0.15 -> ~150 sessions in flight at once; effective context = 512 tokens/turn
- **memory-bound**: KV demand = 10.5 GiB -> 0.33 card (KV is NOT the limiter here -- it fits on one card with room to spare)
- **compute-bound** (M11 anchor): per session 3.2 tok/s (80 tok / 25s) x1000 = 3200 tok/s; A100 ~ 4175 tok/s (835 x 5) -> 0.77 card
- **realistic answer = 0.8 A100 card(s)** for 1000 voice sessions (compute binds, not memory); +1 for HA/SLO headroom -> ~2 in production

## Verdict

A single A100 40GB CANNOT host 1000 concurrent full-context sessions: it tops out at ~112 sessions @2k / ~56 @4k (AWQ weights). Worst-case 1000 needs ~9 cards @2k / ~18 @4k. With a realistic voice duty cycle the limiter shifts from KV memory to compute throughput: ~0.8 card(s) of raw capacity, ~2 with HA/SLO headroom -- see README blueprint.
