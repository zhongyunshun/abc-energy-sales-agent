"""Tests for common/config.py."""

from pathlib import Path

import pytest

from sales_agent.common.config import DEFAULT_SEED, find_repo_root, load_config


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "stage.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_find_repo_root_locates_pyproject():
    root = find_repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "sales_agent").is_dir()


def test_seed_injected_by_default(tmp_path):
    cfg = load_config(write_yaml(tmp_path, "model: foo\n"), repo_root=tmp_path)
    assert cfg["seed"] == DEFAULT_SEED


def test_explicit_seed_preserved(tmp_path):
    cfg = load_config(write_yaml(tmp_path, "seed: 7\n"), repo_root=tmp_path)
    assert cfg["seed"] == 7


def test_relative_paths_resolved_against_repo_root(tmp_path):
    cfg = load_config(
        write_yaml(tmp_path, "input_path: data/raw/x.jsonl\noutput_dir: reports/stage\n"),
        repo_root=tmp_path,
    )
    assert cfg["input_path"] == str((tmp_path / "data/raw/x.jsonl").resolve())
    assert cfg["output_dir"] == str((tmp_path / "reports/stage").resolve())


def test_absolute_path_untouched(tmp_path):
    abs_path = str(tmp_path / "somewhere" / "x.jsonl")
    cfg = load_config(write_yaml(tmp_path, f"input_path: '{abs_path}'\n"), repo_root=tmp_path)
    assert cfg["input_path"] == abs_path


def test_nested_and_listed_paths_resolved(tmp_path):
    text = (
        "sources:\n"
        "  - path: data/a.jsonl\n"
        "  - path: data/b.jsonl\n"
        "train:\n"
        "  output_dir: models/sft\n"
    )
    cfg = load_config(write_yaml(tmp_path, text), repo_root=tmp_path)
    assert cfg["sources"][0]["path"] == str((tmp_path / "data/a.jsonl").resolve())
    assert cfg["sources"][1]["path"] == str((tmp_path / "data/b.jsonl").resolve())
    assert cfg["train"]["output_dir"] == str((tmp_path / "models/sft").resolve())


def test_non_path_keys_left_alone(tmp_path):
    cfg = load_config(
        write_yaml(tmp_path, "model: qwen/qwen3-4b\nnote: data/raw\n"), repo_root=tmp_path
    )
    assert cfg["model"] == "qwen/qwen3-4b"
    assert cfg["note"] == "data/raw"


def test_empty_config_gives_seed_only(tmp_path):
    cfg = load_config(write_yaml(tmp_path, ""), repo_root=tmp_path)
    assert cfg == {"seed": DEFAULT_SEED}


def test_non_mapping_root_rejected(tmp_path):
    with pytest.raises(TypeError, match="mapping"):
        load_config(write_yaml(tmp_path, "- a\n- b\n"), repo_root=tmp_path)
