"""Tests for common/manifest.py."""

import hashlib
import json
from datetime import datetime

from sales_agent.common.manifest import (
    build_manifest,
    get_git_commit,
    sha256_file,
    write_manifest,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world" * 1000)
    assert sha256_file(path) == hashlib.sha256(b"hello world" * 1000).hexdigest()


def test_get_git_commit_in_this_repo():
    commit = get_git_commit()
    assert commit is not None
    assert len(commit) == 40


def test_get_git_commit_outside_repo(tmp_path):
    assert get_git_commit(repo_root=tmp_path) is None


def test_build_manifest_contents(tmp_path):
    f1 = tmp_path / "in1.jsonl"
    f1.write_text('{"a": 1}\n', encoding="utf-8")
    manifest = build_manifest(
        inputs=[f1],
        config={"seed": 42, "model": "foo"},
        stats={"n_records": 1},
    )
    assert manifest["config"] == {"seed": 42, "model": "foo"}
    assert manifest["stats"] == {"n_records": 1}
    assert manifest["git_commit"] is not None
    # timestamp is parseable ISO-8601 with timezone
    ts = datetime.fromisoformat(manifest["created_at"])
    assert ts.tzinfo is not None
    (entry,) = manifest["inputs"]
    assert entry["path"] == str(f1)
    assert entry["sha256"] == sha256_file(f1)
    assert entry["size_bytes"] == f1.stat().st_size


def test_build_manifest_defaults():
    manifest = build_manifest()
    assert manifest["inputs"] == []
    assert manifest["config"] == {}
    assert manifest["stats"] == {}


def test_write_manifest_creates_dir_and_valid_json(tmp_path):
    out_dir = tmp_path / "stage_out"
    path = write_manifest(out_dir, build_manifest(stats={"n": 5}))
    assert path == out_dir / "manifest.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["stats"] == {"n": 5}
