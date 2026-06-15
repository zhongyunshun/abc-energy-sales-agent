"""YAML config loading with repo-root-relative path resolution (the CLI contract).

Conventions:
- Every stage has one YAML file under ``configs/``.
- Relative paths in configs are resolved against the repository root.
- Every config carries a ``seed`` (default 42) for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEED = 42

# Keys whose string values are treated as paths and resolved to absolute paths.
_PATH_KEY_SUFFIXES = ("path", "dir", "file")


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) to the directory holding pyproject.toml."""
    cur = Path(start) if start else Path(__file__)
    cur = cur.resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml found above {cur}")


def _is_path_key(key: str) -> bool:
    key = key.lower()
    return any(
        key == s or key.endswith("_" + s) or key.endswith(s + "s") for s in _PATH_KEY_SUFFIXES
    )


def _resolve_paths(node: Any, root: Path, key: str = "") -> Any:
    """Recursively resolve relative path strings against the repo root.

    A string is treated as a path when its dict key looks path-like
    (``path``/``dir``/``file`` suffix, singular or plural). Absolute paths
    are left untouched.
    """
    if isinstance(node, dict):
        return {k: _resolve_paths(v, root, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v, root, key) for v in node]
    if isinstance(node, str) and _is_path_key(key):
        p = Path(node)
        return str(p) if p.is_absolute() else str((root / p).resolve())
    return node


def load_config(path: str | Path, repo_root: str | Path | None = None) -> dict:
    """Load a stage YAML config.

    - Resolves relative path-like values against the repo root.
    - Injects ``seed: 42`` when absent.
    - Returns a plain dict (stage-specific typed models are each module's concern).
    """
    path = Path(path)
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}: {path}")
    cfg = _resolve_paths(raw, root)
    cfg.setdefault("seed", DEFAULT_SEED)
    return cfg
