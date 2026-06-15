"""GPU integration smoke for M6 (T6.4). Deselected by default; run inside the
train env with: ``uv run pytest -m gpu`` (or the chippie06 venv).

Validates the merge script runs end-to-end on a tiny --smoke config against the
real M5 DPO adapter and emits a loadable dense HF model (safetensors + tokenizer +
chat template) plus a manifest whose ``consistency_check`` verdict is recorded.
This exercises the real BF16 merge_and_unload + the PEFT-vs-merged greedy
comparison against the actual stack.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "merge.yaml"
SCRIPT = REPO_ROOT / "scripts" / "training" / "merge_adapter.py"
DPO_ADAPTER = REPO_ROOT / "models" / "adapters" / "dpo"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m6_merge_adapter_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_merge_produces_loadable_dense_model(tmp_path):
    if not (DPO_ADAPTER / "adapter_config.json").exists():
        pytest.skip(f"DPO adapter not present at {DPO_ADAPTER}")

    cli = _load_cli_module()
    out = tmp_path / "merged"
    rc = cli.main(
        ["--config", str(CONFIG), "--adapter", str(DPO_ADAPTER),
         "--output-dir", str(out), "--smoke"]
    )
    assert rc == cli.EXIT_OK

    # Dense HF model: safetensors (single or sharded) + tokenizer + chat template.
    has_weights = (out / "model.safetensors").exists() or list(out.glob("model-*.safetensors"))
    assert has_weights
    assert (out / "config.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert (out / "chat_template.jinja").exists()
    # The consistency report travels with the model dir (smoke writes only here,
    # never clobbering the committed reports/training/ evidence).
    assert (out / "consistency_report.md").exists()

    # Manifest records the consistency verdict (DoD).
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    cc = manifest["stats"]["consistency_check"]
    assert cc["consistent"] is True
    assert cc["n_mismatch"] == 0
    assert manifest["stats"]["merge_dtype"] == "bfloat16"  # bf16 merge, not 4-bit

    # transformers can load the merged model and generate a single token.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out))
    model = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16, device_map="auto")
    inputs = tok("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert gen.shape[1] > inputs["input_ids"].shape[1]
