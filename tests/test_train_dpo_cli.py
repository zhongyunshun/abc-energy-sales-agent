"""Host-side contract tests for scripts/training/train_dpo.py (no GPU).

All input-contract checks (missing preference file, unparseable pairs, missing SFT
adapter) run BEFORE the lazy GPU import, so they are exercisable on a CPU-only
host and must return exit code 2 without touching CUDA.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "training" / "train_dpo.py"
REAL_CONFIG = REPO_ROOT / "configs" / "dpo.yaml"
FIXTURES = Path(__file__).parent / "fixtures"


def _load_cli_module():
    # scripts/ is not a package; load by path (same pattern as test_train_sft_gpu).
    spec = importlib.util.spec_from_file_location("m5_train_dpo_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    out = tmp_path / "dpo_test.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def test_missing_pref_file_exits_contract(tmp_path):
    cli = _load_cli_module()
    cfg = _write_config(tmp_path, **{"data.pref_path": "/nonexistent/preference_pairs.jsonl"})
    assert cli.main(["--config", str(cfg)]) == cli.EXIT_CONTRACT


def test_unparseable_pairs_exit_contract(tmp_path):
    # The invalid-pairs fixture is wrapped ({"reason","record"}), so none validate.
    cli = _load_cli_module()
    cfg = _write_config(
        tmp_path, **{"data.pref_path": str(FIXTURES / "preference_pairs_invalid.jsonl")}
    )
    assert cli.main(["--config", str(cfg)]) == cli.EXIT_CONTRACT


def test_missing_sft_adapter_exits_contract(tmp_path):
    # Valid pairs, but the SFT adapter dir has no adapter_config.json -> exit 2
    # (must NOT silently fall back to base for DPO).
    cli = _load_cli_module()
    empty_adapter = tmp_path / "no_adapter"
    empty_adapter.mkdir()
    cfg = _write_config(
        tmp_path,
        **{
            "data.pref_path": str(FIXTURES / "preference_pairs_valid.jsonl"),
            "sft_adapter": str(empty_adapter),
        },
    )
    assert cli.main(["--config", str(cfg)]) == cli.EXIT_CONTRACT


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
