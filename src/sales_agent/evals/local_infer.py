"""Local transformers inference fallback for offline eval (design doc 3-M9 step 2/4, T9.4).

The endpoint path (``generate.py``) is preferred; this is the no-server fallback so
M9 can score a base model or an un-deployed adapter directly with transformers, and
optionally compute gold-reply perplexity. ``torch`` / ``transformers`` / ``peft``
are imported LAZILY inside the functions so this module imports on the CPU-only host
and the prompt-building helper (:func:`build_local_prompt`) stays unit-testable
without the GPU stack.

The prompt string reuses ``training.formatting.render_chatml`` so the local path
renders exactly the unit-tested Qwen ChatML the rest of the pipeline uses
(``add_generation_prompt=True`` -> the reply continues after
``<|im_start|>assistant``). The generated text still goes through
``samples.strip_reasoning`` in the CLI before scoring.

Real model loading / generation / PPL run only on a GPU host (manual / @pytest.mark.gpu),
so they are not part of the host unit-test suite.
"""

from __future__ import annotations

from collections.abc import Sequence

from sales_agent.common.schema import Message
from sales_agent.evals.generate import GenConfig, GenOutput
from sales_agent.evals.samples import EvalSample
from sales_agent.training.formatting import render_chatml


def build_local_prompt(sample: EvalSample, default_system: str | None = None) -> str:
    """Render a sample's prompt to Qwen ChatML text with a generation prompt.

    Mirrors the SFT/serve prompt format via the shared
    :func:`render_chatml`; injects ``default_system`` at position 0 only when the
    sample has no system message (keeps the eval prompt distribution consistent
    with training, like ``to_conversation`` / ``preference_pair_to_dpo``).
    """
    messages = [Message(role=m["role"], content=m["content"]) for m in sample.prompt_messages]
    if default_system is not None and (not messages or messages[0].role != "system"):
        messages = [Message(role="system", content=default_system), *messages]
    return render_chatml(messages, add_generation_prompt=True)


def _load_model(model_path: str, adapter: str | None = None, dtype: str = "bfloat16") -> tuple:
    """Load tokenizer + model (optionally with a PEFT adapter). GPU host only."""
    import torch  # noqa: F401  (selects device/dtype below)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def local_generate(
    samples: Sequence[EvalSample],
    cfg: GenConfig,
    *,
    model_path: str,
    adapter: str | None = None,
    default_system: str | None = None,
    dtype: str = "bfloat16",
) -> list[GenOutput]:
    """Greedy-generate a reply per sample with a local model (GPU host only).

    Deterministic decoding (``do_sample=False``) mirrors the endpoint's
    ``temperature=0``. Returns the same :class:`GenOutput` shape as the endpoint
    path so the CLI scores both identically. ``usage_completion_tokens`` is the
    number of newly generated tokens.
    """
    import torch

    tokenizer, model = _load_model(model_path, adapter=adapter, dtype=dtype)
    outputs: list[GenOutput] = []
    for sample in samples:
        prompt = build_local_prompt(sample, default_system=default_system)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=cfg.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs.append(GenOutput(content=text, usage_completion_tokens=int(new_tokens.shape[0])))
    return outputs


def compute_ppl(
    samples: Sequence[EvalSample],
    *,
    model_path: str,
    adapter: str | None = None,
    default_system: str | None = None,
    dtype: str = "bfloat16",
) -> list[float | None]:
    """Per-sample perplexity of the gold reply under the model (GPU host only).

    Optional metric (design doc 3-M9 step 4): only the gold-reply tokens
    contribute to the loss (the prompt is masked), so this measures how well the
    model predicts the reference continuation. Returns one PPL per sample.
    """
    import math

    import torch

    tokenizer, model = _load_model(model_path, adapter=adapter, dtype=dtype)
    ppls: list[float | None] = []
    for sample in samples:
        prompt = build_local_prompt(sample, default_system=default_system)
        full = prompt + sample.gold
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = tokenizer(full, return_tensors="pt")["input_ids"].to(model.device)
        labels = full_ids.clone()
        labels[0, : prompt_ids.shape[1]] = -100  # mask the prompt; score gold only
        with torch.no_grad():
            loss = model(full_ids, labels=labels).loss
        ppls.append(float(math.exp(loss.item())) if loss is not None else None)
    return ppls
