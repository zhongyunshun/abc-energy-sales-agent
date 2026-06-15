"""M4 T4.6 evidence: greedy base-vs-SFT generations on N held-out val prompts.

For each sampled val dialogue, the context up to (excluding) the final assistant
turn becomes the prompt; we greedily generate once from the base model and once
from base+SFT-adapter, and write a side-by-side markdown for manual inspection of
the style/behaviour shift. GPU; runs in the train venv/container.

Usage:
    uv run python scripts/training/compare_base_sft.py --config configs/sft_a100.yaml \
        --adapter models/adapters/sft --n 5 --out reports/training/sft_base_vs_sft.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import DialogueRecord
from sales_agent.training.formatting import to_conversation


def build_probe(record: DialogueRecord) -> tuple[list[dict], str] | None:
    """Context (messages up to, excluding, the last assistant turn) + gold reply.

    Returns None if there is no assistant turn to hold out.
    """
    msgs = to_conversation(record)["messages"]
    last_assistant = max(
        (i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None
    )
    if last_assistant is None or last_assistant == 0:
        return None
    return msgs[:last_assistant], msgs[last_assistant]["content"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--val", default=None, help="override config data.val_path")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--out", default="reports/training/sft_base_vs_sft.md")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    val_path = Path(args.val or cfg["data"]["val_path"])

    # Evenly sample N probes across the val file (deterministic, no RNG).
    records = [DialogueRecord.model_validate(r) for r in read_jsonl(val_path)]
    probes = []
    for rec in records:
        pr = build_probe(rec)
        if pr is not None:
            probes.append(pr)
    if not probes:
        print("no usable probes in val", file=sys.stderr)
        return 2
    stride = max(1, len(probes) // args.n)
    probes = probes[::stride][: args.n]

    import torch
    from unsloth import FastLanguageModel

    def generate(model_name: str, contexts: list[list[dict]]) -> list[str]:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg["model"]["max_seq_length"],
            load_in_4bit=cfg["model"]["load_in_4bit"],
            dtype=cfg["model"].get("dtype"),
        )
        FastLanguageModel.for_inference(model)
        outs = []
        for ctx in contexts:
            prompt = tokenizer.apply_chat_template(
                ctx, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(text.strip())
        del model
        torch.cuda.empty_cache()
        return outs

    contexts = [c for c, _ in probes]
    base_out = generate(cfg["model"]["name"], contexts)
    sft_out = generate(args.adapter, contexts)

    lines = [
        f"# M4 base vs SFT — {len(probes)} held-out val probes (greedy)\n",
        f"Base: `{cfg['model']['name']}` | SFT adapter: `{args.adapter}` | "
        f"r{cfg['lora']['r']}/alpha{cfg['lora']['alpha']}, "
        f"lr{cfg['train']['learning_rate']:.0e}.\n",
    ]
    for i, (ctx, gold) in enumerate(probes):
        last_user = next((m["content"] for m in reversed(ctx) if m["role"] == "user"), "")
        sysflag = "yes" if ctx[0]["role"] == "system" else "no"
        lines.append(f"\n## Probe {i + 1} (system={sysflag}, turns={len(ctx)})\n")
        lines.append(f"**Last user:** {last_user}\n")
        lines.append(f"**Gold (held-out):** {gold}\n")
        lines.append(f"**BASE:** {base_out[i]}\n")
        lines.append(f"**SFT:** {sft_out[i]}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} ({len(probes)} probes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
