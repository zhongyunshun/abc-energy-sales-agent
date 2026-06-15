#!/usr/bin/env python3
"""Bonus 1 -- A100 40GB KV-cache capacity analysis for Qwen3-4B (AWQ INT4).

Pure arithmetic, no GPU required. Answers: "can a single A100 40GB host 1000
concurrent sessions?" by sizing the PagedAttention KV cache against the card.

All structural parameters are READ from the real served model
(models/quantized/awq/config.json), and weight sizes from the M7 quant manifest
(reports/training/quant_manifest.json) -- nothing about the model is hard-coded,
so the numbers track whatever was actually quantized and deployed.

Run:
    python Bonus/bonus1_capacity/kv_capacity_calc.py
Writes a Markdown table + a JSON sidecar next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

# --- locations (repo-relative; this file lives in Bonus/bonus1_capacity/) ------
REPO = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPO / "models" / "quantized" / "awq" / "config.json"
QUANT_MANIFEST = REPO / "reports" / "training" / "quant_manifest.json"
OUT_JSON = Path(__file__).resolve().parent / "kv_capacity.json"
OUT_MD = Path(__file__).resolve().parent / "kv_capacity_table.md"

# --- assumptions (documented; tweak here, everything downstream follows) -------
GiB = 1024**3
A100_PHYSICAL_GIB = 40.0          # NVIDIA A100 40GB (nominal; ~39.5 GiB really usable)
GPU_MEM_UTIL = 0.90               # vLLM --gpu-memory-utilization on a dedicated card
NON_KV_OVERHEAD_GIB = 2.0         # CUDA context + activations + CUDA-graph capture (vLLM)
KV_DTYPE_BYTES = 2                # KV cache stored in bf16/fp16 (W4A16 keeps KV in 16-bit)
CONTEXTS = [2048, 4096]           # the 2k / 4k tiers the brief asks for
TARGET_SESSIONS = 1000

# Realistic voice refinement (see README): a session is only mid-generation a
# fraction of wall-time (caller speaking + ASR + TTS + think), and a sales turn
# carries far less than the max window.
VOICE_ACTIVE_FRACTION = 0.15      # ~15% of wall-time actually decoding
VOICE_EFFECTIVE_CTX = 512         # avg live tokens held per in-flight turn

# Throughput anchor (M11 Locust bench, reports/bench/): the 4070/INT4 service
# saturated at ~835 tok/s aggregate (concurrency 16->32 knee). An A100 has far
# more compute; we apply a deliberately CONSERVATIVE multiplier and treat the
# result as the binding constraint in the realistic voice case (compute, not KV).
M11_SAT_TOK_S_4070 = 835.0        # measured plateau on the 4070 (M11 bench_summary.csv)
A100_COMPUTE_MULT = 5.0           # conservative A100-vs-4070 serving-throughput factor
# Per-session sustained generation demand in a voice dialogue.
VOICE_TURN_TOKENS = 80            # output tokens per assistant turn (M8 demo: ~40-55)
VOICE_TURN_INTERVAL_S = 25.0      # one agent turn every ~25s of conversation


def load_model_params() -> dict:
    cfg = json.loads(MODEL_CONFIG.read_text())
    # head_dim is explicit in Qwen3 configs; fall back to hidden/heads otherwise.
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return {
        "model_type": cfg.get("model_type"),
        "num_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_key_value_heads"],   # GQA: KV heads << attn heads
        "head_dim": head_dim,
        "hidden_size": cfg["hidden_size"],
        "max_position_embeddings": cfg.get("max_position_embeddings"),
    }


def load_weight_sizes() -> dict:
    sizes = json.loads(QUANT_MANIFEST.read_text())["stats"]["sizes"]
    return {
        "awq_int4_gib": sizes["int4_bytes"] / GiB,
        "bf16_gib": sizes["fp16_bytes"] / GiB,
    }


def kv_bytes_per_token(p: dict) -> int:
    # K and V, every layer, every KV head: the canonical PagedAttention sizing.
    #   bytes/token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
    return p["num_layers"] * 2 * p["num_kv_heads"] * p["head_dim"] * KV_DTYPE_BYTES


def kv_budget_gib(weight_gib: float) -> float:
    usable = A100_PHYSICAL_GIB * GPU_MEM_UTIL
    return usable - weight_gib - NON_KV_OVERHEAD_GIB


def main() -> None:
    p = load_model_params()
    w = load_weight_sizes()
    per_tok = kv_bytes_per_token(p)
    per_tok_kib = per_tok / 1024

    # KV budget under the two weight footprints we actually have.
    budget_awq = kv_budget_gib(w["awq_int4_gib"])
    budget_bf16 = kv_budget_gib(w["bf16_gib"])

    rows = []
    for ctx in CONTEXTS:
        per_session_gib = per_tok * ctx / GiB
        demand_1000_gib = per_session_gib * TARGET_SESSIONS
        max_sess_awq = int(budget_awq // per_session_gib)
        max_sess_bf16 = int(budget_bf16 // per_session_gib)
        cards_awq = demand_1000_gib / budget_awq
        rows.append({
            "context_tokens": ctx,
            "kv_per_session_mib": per_session_gib * 1024,
            "kv_1000_sessions_gib": demand_1000_gib,
            "max_sessions_awq": max_sess_awq,
            "max_sessions_bf16": max_sess_bf16,
            "a100_cards_for_1000_awq": cards_awq,
        })

    # Realistic voice scenario (AWQ weights): only a fraction in flight, short ctx.
    inflight = int(TARGET_SESSIONS * VOICE_ACTIVE_FRACTION)
    voice_per_session_gib = per_tok * VOICE_EFFECTIVE_CTX / GiB
    voice_demand_gib = voice_per_session_gib * inflight
    voice_cards_mem = voice_demand_gib / budget_awq

    # In the realistic case KV fits easily, so COMPUTE (tok/s) binds, not memory.
    a100_tok_s = M11_SAT_TOK_S_4070 * A100_COMPUTE_MULT
    per_session_tok_s = VOICE_TURN_TOKENS / VOICE_TURN_INTERVAL_S
    agg_tok_s_1000 = per_session_tok_s * TARGET_SESSIONS
    voice_cards_compute = agg_tok_s_1000 / a100_tok_s
    # The realistic answer is whichever constraint binds harder.
    voice_cards = max(voice_cards_mem, voice_cards_compute)

    result = {
        "model_params": p,
        "weight_sizes_gib": w,
        "assumptions": {
            "a100_physical_gib": A100_PHYSICAL_GIB,
            "gpu_mem_util": GPU_MEM_UTIL,
            "non_kv_overhead_gib": NON_KV_OVERHEAD_GIB,
            "kv_dtype_bytes": KV_DTYPE_BYTES,
            "target_sessions": TARGET_SESSIONS,
        },
        "kv_bytes_per_token": per_tok,
        "kv_kib_per_token": per_tok_kib,
        "kv_budget_gib": {"awq_int4": budget_awq, "bf16": budget_bf16},
        "worst_case_full_context": rows,
        "voice_realistic": {
            "active_fraction": VOICE_ACTIVE_FRACTION,
            "effective_ctx_tokens": VOICE_EFFECTIVE_CTX,
            "inflight_sessions": inflight,
            "kv_demand_gib": voice_demand_gib,
            "cards_memory_bound": voice_cards_mem,
            "m11_sat_tok_s_4070": M11_SAT_TOK_S_4070,
            "a100_compute_mult": A100_COMPUTE_MULT,
            "a100_tok_s_est": a100_tok_s,
            "per_session_tok_s": per_session_tok_s,
            "agg_tok_s_1000": agg_tok_s_1000,
            "cards_compute_bound": voice_cards_compute,
            "a100_cards_for_1000_sessions": voice_cards,
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    # ---- human-readable report ----
    lines = []
    lines.append(f"# KV-cache capacity -- {p['model_type']} (AWQ INT4) on A100 40GB\n")
    lines.append("## Model structure (read from models/quantized/awq/config.json)\n")
    lines.append(f"- num_layers = {p['num_layers']}")
    lines.append(f"- num_kv_heads = {p['num_kv_heads']} (GQA; attn heads = {p['num_attention_heads']})")
    lines.append(f"- head_dim = {p['head_dim']}")
    lines.append(f"- KV dtype = {KV_DTYPE_BYTES} bytes (bf16)")
    lines.append(f"- max_position_embeddings = {p['max_position_embeddings']}\n")
    lines.append("## Per-token KV\n")
    lines.append("`bytes/token = num_layers x 2 x num_kv_heads x head_dim x dtype_bytes`")
    lines.append(
        f"= {p['num_layers']} x 2 x {p['num_kv_heads']} x {p['head_dim']} x {KV_DTYPE_BYTES} "
        f"= **{per_tok:,} B = {per_tok_kib:.0f} KiB/token**\n"
    )
    lines.append("## Memory budget on a single A100 40GB\n")
    lines.append(f"- usable @ util {GPU_MEM_UTIL} = {A100_PHYSICAL_GIB * GPU_MEM_UTIL:.1f} GiB")
    lines.append(f"- AWQ INT4 weights = {w['awq_int4_gib']:.2f} GiB  |  bf16 weights = {w['bf16_gib']:.2f} GiB")
    lines.append(f"- non-KV overhead = {NON_KV_OVERHEAD_GIB:.1f} GiB")
    lines.append(f"- **KV budget (AWQ) = {budget_awq:.1f} GiB** | KV budget (bf16) = {budget_bf16:.1f} GiB\n")
    lines.append("## Worst case: 1000 sessions each holding full context\n")
    lines.append("| ctx | KV/session | KV x 1000 | max sessions (AWQ) | max sessions (bf16) | A100s for 1000 (AWQ) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['context_tokens']} | {r['kv_per_session_mib']:.0f} MiB | "
            f"{r['kv_1000_sessions_gib']:.0f} GiB | {r['max_sessions_awq']} | "
            f"{r['max_sessions_bf16']} | {r['a100_cards_for_1000_awq']:.1f} |"
        )
    lines.append("")
    lines.append("## Realistic voice scenario (AWQ weights)\n")
    lines.append(
        f"- active fraction = {VOICE_ACTIVE_FRACTION} -> ~{inflight} sessions in flight at once; "
        f"effective context = {VOICE_EFFECTIVE_CTX} tokens/turn"
    )
    lines.append(
        f"- **memory-bound**: KV demand = {voice_demand_gib:.1f} GiB -> {voice_cards_mem:.2f} card "
        f"(KV is NOT the limiter here -- it fits on one card with room to spare)"
    )
    lines.append(
        f"- **compute-bound** (M11 anchor): per session {per_session_tok_s:.1f} tok/s "
        f"({VOICE_TURN_TOKENS} tok / {VOICE_TURN_INTERVAL_S:.0f}s) x1000 = {agg_tok_s_1000:.0f} tok/s; "
        f"A100 ~ {a100_tok_s:.0f} tok/s ({M11_SAT_TOK_S_4070:.0f} x {A100_COMPUTE_MULT:.0f}) "
        f"-> {voice_cards_compute:.2f} card"
    )
    lines.append(
        f"- **realistic answer = {voice_cards:.1f} A100 card(s)** for 1000 voice sessions "
        f"(compute binds, not memory); +1 for HA/SLO headroom -> ~2 in production\n"
    )

    verdict_2k = rows[0]["max_sessions_awq"]
    lines.append("## Verdict\n")
    lines.append(
        f"A single A100 40GB CANNOT host 1000 concurrent full-context sessions: it tops out at "
        f"~{verdict_2k} sessions @2k / ~{rows[1]['max_sessions_awq']} @4k (AWQ weights). "
        f"Worst-case 1000 needs ~{rows[0]['a100_cards_for_1000_awq']:.0f} cards @2k / "
        f"~{rows[1]['a100_cards_for_1000_awq']:.0f} @4k. With a realistic voice duty cycle the "
        f"limiter shifts from KV memory to compute throughput: ~{voice_cards:.1f} card(s) of raw "
        f"capacity, ~2 with HA/SLO headroom -- see README blueprint."
    )
    OUT_MD.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[written] {OUT_JSON}")
    print(f"[written] {OUT_MD}")


if __name__ == "__main__":
    main()
