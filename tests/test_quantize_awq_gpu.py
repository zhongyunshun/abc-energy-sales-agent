"""GPU integration smoke for M7 (T7.2/T7.3). Deselected by default; run inside the
train container with: ``uv run pytest -m gpu``.

Validates quantize_awq.py runs end-to-end on a tiny --smoke calibration set
against the real M6 merged model and emits a loadable compressed-tensors INT4
model (safetensors + tokenizer + chat template) plus a manifest recording the AWQ
recipe, FP16-vs-INT4 sizes, and the self-check verdict. Exercises the real
llm-compressor oneshot + transformers reload against the actual stack.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "quant.yaml"
SCRIPT = REPO_ROOT / "scripts" / "quant" / "quantize_awq.py"
MERGED = REPO_ROOT / "models" / "adapters" / "merged"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m7_quantize_awq_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_quant_produces_loadable_int4_model(tmp_path):
    if not (MERGED / "config.json").exists():
        pytest.skip(f"merged model not present at {MERGED}")

    cli = _load_cli_module()
    out = tmp_path / "awq"
    rc = cli.main(
        ["--config", str(CONFIG), "--merged-dir", str(MERGED),
         "--output-dir", str(out), "--smoke"]
    )
    assert rc == cli.EXIT_OK

    # Compressed INT4 model: safetensors (single or sharded) + tokenizer + template.
    has_weights = (out / "model.safetensors").exists() or list(out.glob("model-*.safetensors"))
    assert has_weights
    assert (out / "config.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert (out / "chat_template.jinja").exists()

    # config.json declares a quantization_config (compressed-tensors / AWQ).
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" in cfg

    # Manifest records recipe + sizes + self-check verdict (DoD / T7.3).
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["stats"]
    assert stats["recipe"]["scheme"] == "W4A16_ASYM"
    assert stats["recipe"]["group_size"] == 128
    assert stats["self_check_ok"] is True
    sizes = stats["sizes"]
    assert sizes["int4_bytes"] < sizes["fp16_bytes"]  # INT4 is smaller than FP16
    assert sizes["compression_ratio"] > 1.0

    # transformers can load the quantized model and generate a single token.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out))
    model = AutoModelForCausalLM.from_pretrained(str(out), device_map="auto")
    inputs = tok("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert gen.shape[1] > inputs["input_ids"].shape[1]
