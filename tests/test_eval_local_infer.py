"""Unit tests for the testable (no-GPU) part of the M9 local fallback (T9.4).

Only build_local_prompt is exercised on the host -- model loading / generation /
PPL require torch+transformers and run on a GPU host (manual). This pins that the
local path renders the same Qwen ChatML (with a generation prompt) as SFT/serve.
"""

from __future__ import annotations

from sales_agent.evals.local_infer import build_local_prompt
from sales_agent.evals.samples import EvalSample
from sales_agent.training.formatting import RESPONSE_PART


def _sample(prompt_messages) -> EvalSample:
    return EvalSample(id="d", scenario="general", prompt_messages=prompt_messages, gold="g")


def test_build_local_prompt_has_generation_prompt():
    s = _sample([{"role": "user", "content": "Hi"}])
    text = build_local_prompt(s)
    assert text.endswith(RESPONSE_PART)  # ready for the model to continue
    assert "<|im_start|>user\nHi<|im_end|>" in text


def test_build_local_prompt_injects_default_system():
    s = _sample([{"role": "user", "content": "Hi"}])
    text = build_local_prompt(s, default_system="You are an agent.")
    assert text.startswith("<|im_start|>system\nYou are an agent.<|im_end|>")


def test_build_local_prompt_keeps_existing_system():
    s = _sample([
        {"role": "system", "content": "Custom."},
        {"role": "user", "content": "Hi"},
    ])
    text = build_local_prompt(s, default_system="Should not appear.")
    assert "Custom." in text
    assert "Should not appear." not in text
