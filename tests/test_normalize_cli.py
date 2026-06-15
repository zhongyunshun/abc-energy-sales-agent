"""End-to-end tests for the M1 thin CLI (scripts/data/normalize.py).

Runs the script as a subprocess against local fixture sources only — no
network or HF downloads. Verifies outputs, report counts, and exit codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import DialogueRecord, validate_dialogue

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "data" / "normalize.py"


def write_config(path: Path, output_dir: Path, sources: list[dict]) -> Path:
    cfg = {
        "seed": 42,
        "output_dir": str(output_dir),
        "smoke_limit": 3,
        "sources": sources,
        "clean": {"min_content_chars": 2},
        "scenario_keywords": {
            "objection_handling": ["too expensive", "not interested"],
            "info_gathering": ["how much electricity", "current contract"],
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


def local_sources(workspace: Path) -> list[dict]:
    fixtures = workspace / "fixtures"
    return [
        {
            "source_tag": "local:alpaca-fixture",
            "path": str(fixtures / "raw_alpaca.jsonl"),
            "format": "alpaca",
        },
        {
            "source_tag": "local:sharegpt-fixture",
            "path": str(fixtures / "raw_sharegpt.jsonl"),
            "format": "sharegpt",
        },
    ]


class TestNormalizeCli:
    def test_full_run_writes_valid_outputs(self, tmp_workspace: Path):
        out_dir = tmp_workspace / "out"
        cfg = write_config(tmp_workspace / "cfg.yaml", out_dir, local_sources(tmp_workspace))

        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 0, proc.stderr

        records = [
            DialogueRecord.model_validate(r) for r in read_jsonl(out_dir / "normalized.jsonl")
        ]
        assert len(records) == 8
        assert all(validate_dialogue(r) == [] for r in records)

        report = json.loads((out_dir / "normalize_report.json").read_text(encoding="utf-8"))
        assert report["totals"]["input"] == 19
        assert report["totals"]["output"] == 8
        assert report["sources"]["local:alpaca-fixture"]["dropped"]["conversion_failed"] == 4

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["inputs"]) == 2  # both local sources hashed
        assert manifest["stats"]["output"] == 8

    def test_smoke_mode_caps_records_per_source(self, tmp_workspace: Path):
        out_dir = tmp_workspace / "out"
        cfg = write_config(tmp_workspace / "cfg.yaml", out_dir, local_sources(tmp_workspace))

        proc = run_cli("--config", str(cfg), "--smoke")
        assert proc.returncode == 0, proc.stderr
        report = json.loads((out_dir / "normalize_report.json").read_text(encoding="utf-8"))
        assert report["totals"]["input"] == 6  # smoke_limit=3 per source

    def test_output_dir_override(self, tmp_workspace: Path):
        cfg = write_config(
            tmp_workspace / "cfg.yaml", tmp_workspace / "ignored", local_sources(tmp_workspace)
        )
        override = tmp_workspace / "override"
        proc = run_cli("--config", str(cfg), "--output-dir", str(override))
        assert proc.returncode == 0, proc.stderr
        assert (override / "normalized.jsonl").exists()
        assert not (tmp_workspace / "ignored").exists()

    def test_missing_local_file_exits_2(self, tmp_workspace: Path):
        sources = [
            {
                "source_tag": "local:missing",
                "path": str(tmp_workspace / "nope.jsonl"),
                "format": "alpaca",
            }
        ]
        cfg = write_config(tmp_workspace / "cfg.yaml", tmp_workspace / "out", sources)
        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 2

    def test_no_sources_exits_2(self, tmp_workspace: Path):
        cfg = write_config(tmp_workspace / "cfg.yaml", tmp_workspace / "out", [])
        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 2

    def test_bad_format_exits_2(self, tmp_workspace: Path):
        sources = [
            {
                "source_tag": "local:bad",
                "path": str(tmp_workspace / "fixtures" / "raw_alpaca.jsonl"),
                "format": "csv",
            }
        ]
        cfg = write_config(tmp_workspace / "cfg.yaml", tmp_workspace / "out", sources)
        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 2
