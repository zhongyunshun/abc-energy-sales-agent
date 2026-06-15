"""Async batch generation against an OpenAI-compatible endpoint (design doc 3-M9 step 2).

The orchestration takes an *injected* async client structurally compatible with
``openai.AsyncOpenAI`` (``client.chat.completions.create``), so unit tests drive it
with the programmable fake from ``tests/conftest.py`` -- zero real API calls -- and
production passes a real ``AsyncOpenAI`` pointed at the vLLM endpoint. The thin CLI
owns client construction and file I/O.

Generation parameters are fixed for reproducibility (design doc 3-M9: temperature=0,
fixed max_tokens). We read ``message.content`` ONLY -- never ``reasoning_content`` --
because M8's qwen3 reasoning parser already moved the empty ``<think></think>`` out
of content; the scoring pipeline still strips defensively (see
``samples.strip_reasoning``) for the no-parser local path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sales_agent.evals.samples import EvalSample


@dataclass(frozen=True)
class GenConfig:
    """Fixed generation parameters (from eval_offline.yaml ``generation`` block)."""

    temperature: float = 0.0
    max_tokens: int = 256
    concurrency: int = 16

    @classmethod
    def from_dict(cls, cfg: dict | None) -> GenConfig:
        cfg = cfg or {}
        return cls(
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 256),
            concurrency=cfg.get("concurrency", 16),
        )

    def as_record(self) -> dict:
        """Compact gen-config snapshot for the results/summary contract."""
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


@dataclass(frozen=True)
class GenOutput:
    """One generation result: raw reply content + the endpoint's token usage."""

    content: str
    usage_completion_tokens: int | None
    raw: Any = None


async def generate_one(
    client: Any, model: str, messages: list[dict], cfg: GenConfig
) -> GenOutput:
    """Issue one non-streaming chat completion and extract content + usage.

    Reads ``choices[0].message.content`` (NOT ``reasoning_content``); records
    ``usage.completion_tokens`` when present (reference field -- the canonical
    length metric is recomputed locally for cross-group consistency).
    """
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        stream=False,
    )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    return GenOutput(content=content, usage_completion_tokens=completion_tokens, raw=resp)


async def generate_all(
    samples: Sequence[EvalSample], client: Any, model: str, cfg: GenConfig
) -> list[GenOutput]:
    """Generate a reply for every sample concurrently (bounded by ``concurrency``).

    Order of outputs matches ``samples`` order (``asyncio.gather`` preserves it),
    so the CLI can zip them back together by position.
    """
    sem = asyncio.Semaphore(cfg.concurrency)

    async def worker(sample: EvalSample) -> GenOutput:
        async with sem:
            return await generate_one(client, model, sample.prompt_messages, cfg)

    return list(await asyncio.gather(*(worker(s) for s in samples)))
