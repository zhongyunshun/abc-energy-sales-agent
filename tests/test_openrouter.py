"""Tests for common/openrouter.py using the programmable fake (no real API)."""

import asyncio

import pytest

from sales_agent.common.openrouter import (
    API_KEY_ENV_VAR,
    OpenRouterClient,
    OpenRouterError,
)


class TransientError(Exception):
    """Stands in for a retryable transient failure."""


def make_client(fake, **kwargs):
    kwargs.setdefault("backoff_base", 0)  # instant retries in tests
    kwargs.setdefault("retryable", (TransientError,))
    return OpenRouterClient("fake/model", raw_client=fake, **kwargs)


MESSAGES = [{"role": "user", "content": "Hello"}]


class TestChat:
    async def test_success_returns_content_and_usage(self, fake_openrouter):
        fake_openrouter.queue_response("Hi there!", prompt_tokens=7, completion_tokens=3)
        client = make_client(fake_openrouter)
        result = await client.chat(MESSAGES, temperature=0.9)
        assert result.content == "Hi there!"
        assert result.model == "fake/model"
        assert (result.prompt_tokens, result.completion_tokens) == (7, 3)
        # request passthrough
        assert fake_openrouter.calls[0]["messages"] == MESSAGES
        assert fake_openrouter.calls[0]["temperature"] == 0.9
        assert fake_openrouter.calls[0]["model"] == "fake/model"

    async def test_per_call_model_override(self, fake_openrouter):
        fake_openrouter.queue_response("ok")
        client = make_client(fake_openrouter)
        result = await client.chat(MESSAGES, model="other/model")
        assert result.model == "other/model"
        assert fake_openrouter.calls[0]["model"] == "other/model"

    async def test_retry_then_succeed(self, fake_openrouter):
        fake_openrouter.queue_error(TransientError("boom 1"))
        fake_openrouter.queue_error(TransientError("boom 2"))
        fake_openrouter.queue_response("recovered")
        client = make_client(fake_openrouter, max_retries=3)
        result = await client.chat(MESSAGES)
        assert result.content == "recovered"
        assert fake_openrouter.call_count == 3
        assert client.usage.retries == 2
        assert client.usage.failures == 0

    async def test_retries_exhausted_raises(self, fake_openrouter):
        for i in range(3):
            fake_openrouter.queue_error(TransientError(f"boom {i}"))
        client = make_client(fake_openrouter, max_retries=2)
        with pytest.raises(OpenRouterError, match="after 2 retries"):
            await client.chat(MESSAGES)
        assert fake_openrouter.call_count == 3  # initial + 2 retries
        assert client.usage.failures == 1

    async def test_non_retryable_error_surfaces_immediately(self, fake_openrouter):
        fake_openrouter.queue_error(ValueError("contract bug"))
        client = make_client(fake_openrouter, max_retries=3)
        with pytest.raises(ValueError, match="contract bug"):
            await client.chat(MESSAGES)
        assert fake_openrouter.call_count == 1

    async def test_none_content_normalized_to_empty_string(self, fake_openrouter):
        fake_openrouter.queue_response("placeholder")
        fake_openrouter.script[0].choices[0].message.content = None
        client = make_client(fake_openrouter)
        result = await client.chat(MESSAGES)
        assert result.content == ""


class TestUsageAccounting:
    async def test_usage_accumulates_across_calls(self, fake_openrouter):
        fake_openrouter.queue_response("a", prompt_tokens=10, completion_tokens=5)
        fake_openrouter.queue_response("b", prompt_tokens=20, completion_tokens=15)
        client = make_client(fake_openrouter)
        await client.chat(MESSAGES)
        await client.chat(MESSAGES, model="other/model")
        usage = client.usage
        assert usage.requests == 2
        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 50
        assert usage.requests_by_model == {"fake/model": 1, "other/model": 1}
        summary = usage.as_dict()
        assert summary["total_tokens"] == 50
        assert summary["failures"] == 0


class TestConcurrency:
    async def test_semaphore_caps_in_flight_requests(self, fake_openrouter):
        in_flight = 0
        max_in_flight = 0

        async def slow_create(**kwargs):
            nonlocal in_flight, max_in_flight
            fake_openrouter.calls.append(kwargs)
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            from tests.conftest import make_chat_response

            return make_chat_response("ok")

        fake_openrouter.chat.completions.create = slow_create
        client = make_client(fake_openrouter, concurrency=3)
        await asyncio.gather(*(client.chat(MESSAGES) for _ in range(10)))
        assert fake_openrouter.call_count == 10
        assert max_in_flight <= 3


class TestFakeOpenRouter:
    """The fake itself is shared test infrastructure for M2/M10 — pin its behavior."""

    async def test_script_exhaustion_fails_loudly(self, fake_openrouter):
        client = make_client(fake_openrouter)
        with pytest.raises(AssertionError, match="script exhausted"):
            await client.chat(MESSAGES)

    async def test_default_response_after_script(self, fake_openrouter):
        fake_openrouter.queue_response("scripted")
        fake_openrouter.set_default("default")
        client = make_client(fake_openrouter)
        assert (await client.chat(MESSAGES)).content == "scripted"
        assert (await client.chat(MESSAGES)).content == "default"
        assert (await client.chat(MESSAGES)).content == "default"

    async def test_queue_dynamic_sees_request_kwargs(self, fake_openrouter):
        from tests.conftest import make_chat_response

        fake_openrouter.queue_dynamic(
            lambda kw: make_chat_response(f"echo: {kw['messages'][0]['content']}")
        )
        client = make_client(fake_openrouter)
        result = await client.chat([{"role": "user", "content": "ping"}])
        assert result.content == "echo: ping"

    async def test_queue_dynamic_can_inject_error(self, fake_openrouter):
        fake_openrouter.queue_dynamic(lambda kw: TransientError("dynamic boom"))
        fake_openrouter.queue_response("ok")
        client = make_client(fake_openrouter)
        result = await client.chat(MESSAGES)
        assert result.content == "ok"
        assert client.usage.retries == 1

    async def test_bad_output_passthrough_for_downstream_validation(self, fake_openrouter):
        """M2-style usage: the fake returns malformed payloads verbatim."""
        fake_openrouter.queue_responses(
            "not json at all",
            '{"messages": []}',
            '{"messages": [{"role": "assistant", "content": "only $0.09 per kWh"}]}',
        )
        client = make_client(fake_openrouter)
        outputs = [(await client.chat(MESSAGES)).content for _ in range(3)]
        assert outputs[0] == "not json at all"
        assert "$0.09" in outputs[2]


class TestApiKeyHandling:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        with pytest.raises(OpenRouterError, match=API_KEY_ENV_VAR):
            OpenRouterClient("fake/model")

    def test_env_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, "sk-test-123")
        client = OpenRouterClient("fake/model")
        assert client.model == "fake/model"
