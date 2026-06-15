"""Stage output manifests for reproducibility (the module-boundary contract, principle 5).

Every stage writes a ``manifest.json`` into its output directory containing:
input file hashes, a config snapshot, the current git commit, a UTC timestamp,
and stage-specific key statistics.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(repo_root: str | Path | None = None) -> str | None:
    """Current HEAD commit hash, or None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def build_manifest(
    inputs: list[str | Path] | None = None,
    config: dict | None = None,
    stats: dict | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Assemble a manifest dict for a stage run.

    ``inputs`` are existing files; each is recorded with its size and SHA-256
    so downstream consumers can verify exactly what a stage was run on.
    """
    input_entries = []
    for p in inputs or []:
        p = Path(p)
        input_entries.append(
            {"path": str(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(repo_root),
        "inputs": input_entries,
        "config": config or {},
        "stats": stats or {},
    }


def write_manifest(output_dir: str | Path, manifest: dict) -> Path:
    """Write manifest.json into ``output_dir`` (created if missing)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path
