"""GPU integration smoke for M5 (T5.4). Deselected by default; run inside the
train env with: ``uv run pytest -m gpu`` (or the chippie06 venv).

Validates the DPO script runs end-to-end on a tiny --smoke config and emits a
loadable, standalone PEFT adapter at the output_dir root + manifest + curves +
behaviour diff. This is where the exact TRL 0.23.0 DPOConfig/DPOTrainer kwargs and
the two-adapter (ref=SFT) reference path are exercised against the real stack.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "dpo.yaml"
SCRIPT = REPO_ROOT / "scripts" / "training" / "train_dpo.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m5_train_dpo_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_dpo_produces_adapter(tmp_path):
    cli = _load_cli_module()
    out = tmp_path / "dpo"
    rc = cli.main(["--config", str(CONFIG), "--smoke", "--output-dir", str(out)])
    assert rc == cli.EXIT_OK

    # Standalone, M6-mergeable adapter at the output_dir ROOT (not a subdir).
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()

    # Manifest records the DPO provenance and a real margin.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["stats"]
    assert stats["ref_scheme"] == "two_adapter"
    assert stats["base_load_in_4bit"] is False
    assert stats["global_steps"] >= 1
    assert stats["final_reward_margin"] is not None

    # Curves + behaviour diff are produced (report material).
    for art in ("dpo_loss.png", "dpo_margins.png", "dpo_behavior_diff.md"):
        p = REPO_ROOT / "reports" / "training" / art
        assert p.exists() and p.stat().st_size > 0

    # Adapter reloads via PEFT at the SFT rank.
    from peft import PeftConfig

    cfg = PeftConfig.from_pretrained(str(out))
    assert cfg.r == 32
