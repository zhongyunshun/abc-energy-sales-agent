"""Unit tests for M9 endpoint generation (T9.3) -- mock client, zero real calls.

Drives the async orchestration with the programmable FakeOpenRouter from conftest
(structurally compatible with AsyncOpenAI). Pins: fixed gen params are passed
through, content (not reasoning_content) is read, usage tokens are captured, and
output order matches sample order.
"""

from __future__ import annotations

from types import SimpleNamespace

from sales_agent.evals.generate import GenConfig, generate_all, generate_one
from sales_agent.evals.samples import EvalSample


def _sample(rid: str) -> EvalSample:
    return EvalSample(
        id=rid, scenario="general",
        prompt_messages=[{"role": "user", "content": f"q-{rid}"}], gold="g",
    )


def _resp_with_reasoning(content: str, reasoning: str, completion_tokens: int = 7):
    """A response whose message carries BOTH content and reasoning_content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=reasoning))],
        usage=SimpleNamespace(completion_tokens=completion_tokens),
        model="fake",
    )


async def test_generate_one_passes_fixed_params(fake_openrouter):
    fake_openrouter.queue_response("hello")
    cfg = GenConfig(temperature=0.0, max_tokens=256, concurrency=4)
    out = await generate_one(fake_openrouter, "m", [{"role": "user", "content": "hi"}], cfg)
    assert out.content == "hello"
    call = fake_openrouter.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 256
    assert call["model"] == "m"
    assert call["stream"] is False
    assert call["messages"] == [{"role": "user", "content": "hi"}]


async def test_generate_one_reads_content_not_reasoning(fake_openrouter):
    # message has reasoning_content set to a DIFFERENT value -- must be ignored.
    fake_openrouter.script.append(_resp_with_reasoning("clean answer", "secret reasoning"))
    cfg = GenConfig()
    out = await generate_one(fake_openrouter, "m", [{"role": "user", "content": "hi"}], cfg)
    assert out.content == "clean answer"
    assert "reasoning" not in out.content


async def test_generate_one_captures_usage(fake_openrouter):
    fake_openrouter.queue_response("hi", completion_tokens=42)
    out = await generate_one(fake_openrouter, "m", [{"role": "user", "content": "x"}], GenConfig())
    assert out.usage_completion_tokens == 42


async def test_generate_one_null_content_becomes_empty(fake_openrouter):
    fake_openrouter.script.append(SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(completion_tokens=0), model="fake"))
    out = await generate_one(fake_openrouter, "m", [{"role": "user", "content": "x"}], GenConfig())
    assert out.content == ""


async def test_generate_all_preserves_order(fake_openrouter):
    samples = [_sample("a"), _sample("b"), _sample("c")]
    fake_openrouter.queue_responses("ra", "rb", "rc")
    cfg = GenConfig(concurrency=2)
    outs = await generate_all(samples, fake_openrouter, "m", cfg)
    assert [o.content for o in outs] == ["ra", "rb", "rc"]
    assert fake_openrouter.call_count == 3


def test_genconfig_from_dict_defaults():
    c = GenConfig.from_dict(None)
    assert c.temperature == 0.0
    assert c.max_tokens == 256
    assert c.concurrency == 16


def test_genconfig_as_record():
    assert GenConfig(temperature=0.0, max_tokens=128).as_record() == {
        "temperature": 0.0, "max_tokens": 128}
