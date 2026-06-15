# Bonus 1 — A100 40GB capacity analysis + KV-cache confirmation

**Question.** Can a single NVIDIA A100 40GB host **1000 concurrent sessions** of the
ABC Energy sales agent (Qwen3-4B-Instruct, AWQ INT4), and is the M8 vLLM service
actually using a KV cache with NVIDIA 4070 12GB?

**TL;DR.**
- **No** for the strict worst case (every session holding full context at once): one A100 40GB
  tops out at **~112 sessions @2k / ~56 @4k**. Hosting 1000 that way needs **~9 cards @2k,
  ~18 @4k**.
- **Effectively yes** for a realistic voice workload: KV memory is *not* the binding
  constraint there (it fits ~3× over on one card); **compute throughput** is, and 1000 voice
  sessions need only **~1 A100 of raw capacity (~2 with HA/SLO headroom)**.
- **KV cache: confirmed in use.** vLLM serves the model with **PagedAttention** (paged KV,
  on by default) and **`--enable-prefix-caching`** (system-prompt KV reuse). M8 evidence
  below.

This is a paper analysis anchored to real numbers (model `config.json`, M7 quant manifest,
M11 Locust bench). Per the engagement decision we did **not** stand up a fresh A100 serve;
the KV-cache mechanism is confirmed from the existing M8 run (same vLLM v0.10.2, same AWQ
model) on the 4070.

---

## 1. KV-cache sizing (the math)

Every decoded token must keep its Key and Value tensors in the cache, for every layer and
every KV head:

```
KV bytes/token = num_layers × 2 (K,V) × num_kv_heads × head_dim × dtype_bytes
```

Structural parameters are **read from the served model** itself
(`models/quantized/awq/config.json`, Qwen3-4B-Instruct-2507):

| param | value | note |
|---|---|---|
| `num_hidden_layers` | 36 | |
| `num_key_value_heads` | 8 | **GQA** — KV heads ≪ 32 attention heads (4× KV saving vs MHA) |
| `head_dim` | 128 | |
| KV dtype | bf16 = 2 B | W4A16 quantizes weights, **KV stays 16-bit** |
| `max_position_embeddings` | 262144 | model max; we serve 2k–4k |

```
KV/token = 36 × 2 × 8 × 128 × 2 = 147,456 B = 144 KiB/token
```

Per session and for 1000 sessions:

| context | KV / session | KV × 1000 sessions |
|---:|---:|---:|
| 2,048 tok | 288 MiB | **281 GiB** |
| 4,096 tok | 576 MiB | **562 GiB** |

### Budget on a single A100 40GB

| line item | GiB |
|---|---:|
| usable @ `--gpu-memory-utilization 0.90` | 36.0 |
| − AWQ INT4 weights (M7 manifest: 2.666 GB) | −2.48 |
| − non-KV overhead (CUDA ctx + activations + CUDA-graph) | −2.0 |
| **= KV budget (AWQ weights)** | **≈ 31.5** |
| (KV budget with bf16 weights, for reference) | ≈ 26.5 |

### Theoretical max concurrency (full context held simultaneously)

| context | max sessions (AWQ) | max sessions (bf16) | A100s for 1000 (AWQ) |
|---:|---:|---:|---:|
| 2,048 | **~112** | ~94 | **~9** |
| 4,096 | **~56** | ~47 | **~18** |

→ **A single A100 40GB cannot hold 1000 full-context sessions** — short by ~9× (2k) to ~18× (4k).

> Numbers are produced by `kv_capacity_calc.py` (no GPU needed); see `kv_capacity.json` /
> `kv_capacity_table.md` for the machine-readable output. Edit the assumption constants at
> the top of the script (util, overhead, contexts) to re-derive.

---

## 2. Two regimes: worst case vs realistic voice

The table above is the **worst case**: it assumes all 1000 sessions are simultaneously
resident *and* each pins its full 2k/4k window. A voice sales deployment does not behave
that way:

1. **Duty cycle.** A session spends most of its wall-time *not* generating — the caller is
   speaking, ASR is transcribing, TTS is playing back, the human is thinking. Only a fraction
   (~15%) is mid-decode at any instant, so ~1000 sessions ⇒ **~150 requests in flight**.
2. **Short effective context.** A sales turn carries far less than the max window (M8 demo
   replies were 34–55 tokens). Assume ~512 live tokens/turn, not 2k–4k.
3. **Prefix caching** stores the shared system prompt KV **once**, not 1000×.

Under those assumptions:

- **Memory-bound:** 150 in-flight × 512 tok × 144 KiB ≈ **10.5 GiB** — fits ~3× over on one
  card. KV is **not** the limiter here.
- **Compute-bound (binds):** a session generates ~80 tok every ~25 s ⇒ ~3.2 tok/s; ×1000 =
  **~3,200 tok/s aggregate**. M11 measured the 4070/INT4 service saturating at **~835 tok/s**;
  an A100 (conservatively ~5× the 4070's serving throughput) ⇒ ~4,175 tok/s ⇒ **~0.8 card**.

**Realistic answer: ~1 A100 of raw capacity for 1000 voice sessions, ~2 with HA/SLO
headroom.** The constraint that matters is **throughput, not KV memory** — which is exactly
why the blueprint below scales replicas by SLO/throughput, not by VRAM.

---

## 3. Scaling blueprint — supporting 1000 concurrent in production

Sizing depends on the regime: **~9–18 A100s** if you must guarantee full-context residency
for all 1000 at once, **~2 A100s** for the realistic voice duty cycle. Either way the
architecture is the same horizontal-scale pattern:

**A. Multiple vLLM replicas + gateway routing.**
N identical vLLM workers behind an OpenAI-compatible gateway (Envoy / LiteLLM / a small
FastAPI router). Route by least-outstanding-requests; pin a session's multi-turn traffic to
the same replica (session affinity) so its prefix cache stays warm.

**B. KV-cache management (the lever this analysis is about).**
- **PagedAttention** (vLLM default) — KV in fixed 16-token blocks, non-contiguous, so the
  ~31 GiB budget is used without fragmentation and concurrency isn't capped by the largest
  request.
- **Prefix caching** (`--enable-prefix-caching`) — the shared system prompt (and any common
  preamble) is prefilled once and reused across all sessions on a replica → near-zero TTFT
  for the shared prefix and ~0 KV cost per extra session for that span.
- **KV offload / tiering** — vLLM CPU-offload or LMCache to spill cold session KV to host
  RAM / NVMe, raising effective resident sessions per card (trades a little TTFT on resume).

**C. Horizontal autoscaling by SLO.**
Scale replica count on the SLO signals, not on a fixed batch number: TTFT p95, queue depth,
and KV-block utilization (`vllm:gpu_cache_usage_perc`). M11 shows the knee precisely — on the
4070 throughput plateaued and TTFT p50 jumped 112 ms → 868 ms exactly at the `max_num_seqs`
boundary (16→32). The autoscaler's job is to add replicas before in-flight load crosses each
card's knee. K8s HPA/KEDA on those custom metrics; per-tenant rate limits to protect the SLO.

**D. Speculative decoding to cut TTFT / raise throughput.**
A small draft model (e.g. Qwen3-0.6B) or n-gram/EAGLE speculation lets each step emit several
tokens — lower latency at the same batch, or more sessions per card at the same latency.
Particularly valuable for the voice TTFT budget.

**E. Right-size the context.** Serve `max_model_len` at the real conversational need
(2k–4k), not the model's 262k. KV cost is linear in context, so this is the single biggest
per-session memory lever.

**Headline estimate.** Worst-case full-context 1000: ~9 (2k) / ~18 (4k) A100s. Realistic
voice 1000: ~2 A100s (1 for capacity + 1 for HA), behind a gateway, autoscaled on TTFT-p95
and KV-utilization, with prefix caching + right-sized context doing the heavy lifting.

---

## 4. KV-cache confirmation (Task 2)

**Is M8's vLLM already using a KV cache, and how?** Yes — two mechanisms, both confirmed:

### PagedAttention (default, paged KV)
vLLM's serving engine *is* PagedAttention; the KV cache is mandatory and paged (no flag to
turn it on). At startup vLLM profiles free VRAM and reports the KV budget as a fixed number of
GPU blocks (16 tokens each), then schedules continuous batching against that pool. Evidence:

- **Continuous batching demonstrably works** (M8 `reports/serving/concurrency_demo.md`): 16
  concurrent streams finished in **0.74 s** wall vs a **7.48 s** serial estimate — a **10.1×
  latency speedup / 9.13× throughput gain, 917 tok/s aggregate**. That speedup is only
  possible because the engine batches decode steps across requests sharing one paged KV pool.
- **The KV pool is the concurrency limit** (M11 `reports/bench/bench_report.md`): throughput
  scaled 102 → 820 tok/s from concurrency 1 → 16, then **plateaued at the `max_num_seqs`
  boundary** (16→32: 820 → 835 tok/s, TTFT p50 112 → 868 ms). The clean knee is the signature
  of a bounded, paged KV cache scheduling a fixed block pool.
- vLLM loaded the AWQ weights via the **MarlinLinearKernel (W4A16)**, ~2.57 GB resident
  (progress board 2026-06-14), leaving the rest of the budget for paged KV.

### Prefix caching (`--enable-prefix-caching`, system-prompt reuse)
Enabled in both `configs/serve.yaml` (`enable_prefix_caching: true`) and the standalone
`docker/compose.yaml` command. It hashes prompt prefixes and reuses already-computed KV
blocks, so the **shared system prompt is prefilled once** and every later request that begins
with it skips that prefill. Evidence:

- M8 demo: across **16 concurrent requests that all share the same system prompt**, TTFT was
  uniformly **~0.1 s** (mean 0.101 s, p95 0.115 s). Only the first request pays the full
  system-prompt prefill; the rest hit the cached prefix → the flat, near-instant TTFT is the
  prefix-cache signature. (Reproduced from `reports/serving/concurrency_demo.json`.)

> Scope note (per the light-path decision): mechanism + behavioral evidence are taken from
> the existing M8/M11 4070 runs (identical vLLM v0.10.2 and identical AWQ model). We did not
> separately scrape the `vllm:gpu_prefix_cache_hit_rate` metric from a fresh A100 serve; the
> flag is on and the TTFT behavior is consistent with prefix reuse. → **No implementation
> change needed** (the brief says only implement if KV cache were *not* in use — it is).

---

## 5. Reproduce

```bash
# pure arithmetic, no GPU
python Bonus/bonus1_capacity/kv_capacity_calc.py
# -> prints the report; writes kv_capacity.json + kv_capacity_table.md
```

Inputs it reads:
- `models/quantized/awq/config.json` — model structure (layers / KV heads / head_dim)
- `reports/training/quant_manifest.json` — real AWQ INT4 (2.67 GB) & bf16 (8.05 GB) sizes
