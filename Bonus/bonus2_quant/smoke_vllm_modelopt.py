"""Native vLLM smoke test for a ModelOpt unified HF checkpoint.

This is self-contained so it can run in the official vLLM container without the
project package installed. It uses the same five prompts as the M7 quant probe
and writes JSONL evidence under ``Bonus/bonus2_quant/reports``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
RESPONSE_PART = "<|im_start|>assistant\n"


def render_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
    if add_generation_prompt:
        parts.append(RESPONSE_PART)
    return "".join(parts)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompt_rows = read_jsonl(args.prompts)
    prompts = [render_chatml(row["context"], add_generation_prompt=True) for row in prompt_rows]
    baseline_by_id = {}
    if args.baseline and args.baseline.exists():
        baseline_by_id = {row["id"]: row["output"] for row in read_jsonl(args.baseline)}

    llm = LLM(
        model=str(args.model),
        dtype="auto",
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(prompts, sampling)

    rows = []
    for source, result in zip(prompt_rows, outputs, strict=True):
        text = result.outputs[0].text if result.outputs else ""
        rows.append(
            {
                "id": source["id"],
                "native_output": text,
                "fp16_baseline": baseline_by_id.get(source["id"], ""),
                "nonempty": bool(text.strip()),
            }
        )
    write_jsonl(args.output, rows)
    if not all(row["nonempty"] for row in rows):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
