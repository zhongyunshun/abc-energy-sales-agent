"""GPU integration smoke for M4 (T4.3/T4.5). Deselected by default; run inside
the train container with: ``uv run pytest -m gpu``.

Validates that the SFT script runs end-to-end on a tiny --smoke config and emits
a loadable PEFT adapter + manifest + loss curve. This is where the exact
TRL 0.23.0 / Unsloth 2025.11.1 API kwargs (and the train_on_responses_only
masking path) are exercised against the real stack.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "sft.yaml"
SCRIPT = REPO_ROOT / "scripts" / "training" / "train_sft.py"


def _load_cli_module():
    # scripts/ is not a package; load by path (same pattern as test_split_cli).
    spec = importlib.util.spec_from_file_location("m4_train_sft_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_sft_produces_adapter(tmp_path):
    cli = _load_cli_module()
    main, EXIT_OK = cli.main, cli.EXIT_OK

    out = tmp_path / "sft"
    rc = main(["--config", str(CONFIG), "--smoke", "--output-dir", str(out)])
    assert rc == EXIT_OK

    # PEFT adapter is present and reloadable.
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()

    # Manifest records peak VRAM under the 10GB budget and the masking path.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["stats"]
    assert stats["masking"] == "train_on_responses_only"
    assert stats["peak_vram_gb"] <= 10.0, f"peak VRAM {stats['peak_vram_gb']}GB over budget"
    assert stats["global_steps"] >= 1

    # Loss curve PNG was produced (report material).
    loss_png = REPO_ROOT / "reports" / "training" / "sft_loss.png"
    assert loss_png.exists() and loss_png.stat().st_size > 0

    # Adapter reloads via PEFT.
    from peft import PeftConfig

    cfg = PeftConfig.from_pretrained(str(out))
    assert cfg.r == 16
