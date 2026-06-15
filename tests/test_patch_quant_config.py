"""Unit tests for the M8 serve-compat shim's pure stripping logic
(scripts/serving/patch_quant_config.py): it removes ONLY the vLLM-incompatible
quant-config keys, leaves everything else intact, and is idempotent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serving" / "patch_quant_config.py"


def _mod():
    spec = importlib.util.spec_from_file_location("m8_patch_quant_config", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_qc() -> dict:
    # Shape mirrors the real M7 compressed-tensors 0.16.0 quantization_config.
    return {
        "format": "pack-quantized",
        "quant_method": "compressed-tensors",
        "version": "0.16.0",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "targets": ["Linear"],
                "input_activations": None,
                "weights": {
                    "group_size": 128,
                    "num_bits": 4,
                    "symmetric": False,
                    "type": "int",
                    "scale_dtype": None,
                    "zp_dtype": "torch.int8",
                },
            }
        },
    }


def test_strips_only_target_keys():
    m = _mod()
    qc = _sample_qc()
    removed = m.strip_keys(qc, m.INCOMPATIBLE_QUANT_KEYS)
    w = qc["config_groups"]["group_0"]["weights"]
    # target keys gone
    assert "scale_dtype" not in w
    assert "zp_dtype" not in w
    # everything else preserved
    assert w["group_size"] == 128
    assert w["num_bits"] == 4
    assert w["symmetric"] is False
    assert qc["quant_method"] == "compressed-tensors"
    assert qc["version"] == "0.16.0"
    assert qc["config_groups"]["group_0"]["targets"] == ["Linear"]
    # reported paths point at the right place
    assert sorted(removed) == sorted(
        ["config_groups.group_0.weights.scale_dtype",
         "config_groups.group_0.weights.zp_dtype"]
    )


def test_idempotent():
    m = _mod()
    qc = _sample_qc()
    m.strip_keys(qc, m.INCOMPATIBLE_QUANT_KEYS)
    # second pass removes nothing
    assert m.strip_keys(qc, m.INCOMPATIBLE_QUANT_KEYS) == []


def test_noop_on_clean_config():
    m = _mod()
    clean = {"quant_method": "compressed-tensors", "config_groups": {"group_0": {"weights": {}}}}
    assert m.strip_keys(clean, m.INCOMPATIBLE_QUANT_KEYS) == []


def test_handles_lists_and_nesting():
    m = _mod()
    node = {"a": [{"zp_dtype": 1, "keep": 2}, {"scale_dtype": 3}], "b": {"keep": 4}}
    removed = m.strip_keys(node, m.INCOMPATIBLE_QUANT_KEYS)
    assert node == {"a": [{"keep": 2}, {}], "b": {"keep": 4}}
    assert sorted(removed) == ["a[0].zp_dtype", "a[1].scale_dtype"]
