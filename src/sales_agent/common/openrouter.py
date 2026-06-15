"""Unified OpenRouter access layer (the CLI contract).

All strong-model API calls (M2 synthesis, M10 judge) go through
:class:`OpenRouterClient`: the OpenAI SDK pointed at the OpenRouter base URL,
with a concurrency semaphore, exponential-backoff retries, and cumulative
token-usage accounting for cost reports.

The underlying SDK client is injectable (``raw_client``) so unit tests can
substitute the programmable fake from ``tests/conftest.py`` — production code
and tests exercise exactly the same retry/limit/accounting paths.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import openai

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

# Transient failures worth retrying; 4xx contract errors (auth, bad request)
# are not retryable and surface immediately.
DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class OpenRouterError(RuntimeError):
    """Raised when a request fails after exhausting all retries."""


@dataclass
class ChatResult:
    """Normalized result of one chat completion."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: Any = None


@dataclass
class UsageStats:
    """Cumulative usage across a client's lifetime (cost reporting)."""

    requests: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests_by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests_by_model": dict(self.requests_by_model),
        }


class OpenRouterClient:
    """Async chat-completion client with bounded concurrency and retries.

    Args:
        model: default model id (e.g. ``anthropic/claude-sonnet-4-6``).
        api_key: defaults to the ``OPENROUTER_API_KEY`` environment variable.
        concurrency: max in-flight requests (semaphore-bounded).
        max_retries: retry attempts after the first failure.
        backoff_base: first retry delay in seconds; doubles each retry,
            plus up to 25% random jitter. Set 0 in tests for instant retries.
        retryable: exception types that trigger a retry.
        raw_client: injectable SDK-compatible client (tests pass a fake).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        concurrency: int = 8,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        retryable: tuple[type[Exception], ...] = DEFAULT_RETRYABLE,
        raw_client: Any = None,
    ) -> None:
        if raw_client is None:
            api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
            if not api_key:
                raise OpenRouterError(f"no API key: set the {API_KEY_ENV_VAR} environment variable")
            raw_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = raw_client
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.retryable = retryable
        self._semaphore = asyncio.Semaphore(concurrency)
        self.usage = UsageStats()

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run one chat completion with retry; raises OpenRouterError on exhaustion.

        ``kwargs`` (temperature, max_tokens, response_format, ...) are passed
        through to the SDK so each stage config controls its own sampling.
        """
        model = model or self.model
        async with self._semaphore:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                if attempt > 0:
                    delay = min(self.backoff_base * 2 ** (attempt - 1), self.backoff_max)
                    delay *= 1 + random.random() * 0.25
                    logger.info(
                        "retry %d/%d after %.1fs: %s", attempt, self.max_retries, delay, last_exc
                    )
                    self.usage.retries += 1
                    await asyncio.sleep(delay)
                try:
                    response = await self._client.chat.completions.create(
                        model=model, messages=messages, **kwargs
                    )
                except self.retryable as exc:
                    last_exc = exc
                    continue
                return self._record(response, model)
            self.usage.failures += 1
            raise OpenRouterError(
                f"request failed after {self.max_retries} retries: {last_exc}"
            ) from last_exc

    def _record(self, response: Any, model: str) -> ChatResult:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.usage.requests += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.requests_by_model[model] = self.usage.requests_by_model.get(model, 0) + 1
        content = response.choices[0].message.content or ""
        return ChatResult(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=response,
        )
