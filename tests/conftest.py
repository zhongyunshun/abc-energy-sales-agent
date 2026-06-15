"""Shared pytest fixtures (design doc section 4.2).

- ``fake_openrouter``: programmable OpenRouter fake. Downstream module tests
  (M2 synthesis, M10 judge) script it with good outputs, malformed outputs,
  and injected errors — zero real API calls in unit tests.
- ``tmp_workspace``: temp directory pre-populated with contract fixtures so
  module tests never pollute the repo or each other.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_chat_response(
    content: str,
    *,
    model: str = "fake/model",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> SimpleNamespace:
    """Build an object shaped like an OpenAI SDK ChatCompletion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model=model,
    )


@dataclass
class FakeOpenRouter:
    """Programmable stand-in for the OpenAI SDK client used by OpenRouterClient.

    Structurally compatible with ``AsyncOpenAI`` for the
    ``chat.completions.create`` path. Behavior is scripted as a FIFO queue;
    each queued item is consumed by one call and may be:

    - a response (from :func:`make_chat_response` or ``queue_response``),
    - an exception instance (raised — for retry/failure testing),
    - a callable ``(kwargs) -> response`` for dynamic behavior.

    When the script is exhausted, ``default_response`` (if set) is returned;
    otherwise the call fails an assertion, so tests can't silently
    over-consume. Every call's kwargs are recorded in ``calls`` for prompt
    inspection.
    """

    script: list[Any] = field(default_factory=list)
    default_response: SimpleNamespace | None = None
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Mimic the SDK's client.chat.completions.create namespace.
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    # -- scripting API -------------------------------------------------
    def queue_response(self, content: str, **kwargs: Any) -> None:
        """Queue one successful response with the given message content."""
        self.script.append(make_chat_response(content, **kwargs))

    def queue_responses(self, *contents: str) -> None:
        """Queue several successful responses in order."""
        for c in contents:
            self.queue_response(c)

    def queue_error(self, exc: Exception) -> None:
        """Queue an exception to be raised by the next call."""
        self.script.append(exc)

    def queue_dynamic(self, fn: Callable[[dict], Any]) -> None:
        """Queue a callable receiving the request kwargs, returning a response."""
        self.script.append(fn)

    def set_default(self, content: str, **kwargs: Any) -> None:
        """Response returned whenever the script is exhausted."""
        self.default_response = make_chat_response(content, **kwargs)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    # -- SDK-compatible entry point -------------------------------------
    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            if callable(item):
                item = item(kwargs)
                if isinstance(item, Exception):
                    raise item
            return item
        if self.default_response is not None:
            return self.default_response
        raise AssertionError(
            f"FakeOpenRouter script exhausted after {self.call_count} calls "
            "and no default_response set"
        )


@pytest.fixture
def fake_openrouter() -> FakeOpenRouter:
    """A fresh programmable OpenRouter fake per test."""
    return FakeOpenRouter()


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Temp directory with a copy of tests/fixtures/ under ``fixtures/``."""
    dest = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, dest)
    return tmp_path
