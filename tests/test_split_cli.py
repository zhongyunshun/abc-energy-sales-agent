"""End-to-end tests for the M3 thin CLI (scripts/data/split.py).

Runs against tiny generated JSONL inputs (no network, no GPU). Verifies the
happy path (outputs + split_report contract + cross_split_dups == 0), the
downsample wiring, contract-failure exit codes, and -- via an in-process
monkeypatch -- that a detected cross-split leak forces exit code 2 (a real leak
is structurally unreachable through the CLI because global near-dedup runs
before the split, which is exactly the guarantee under test).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from sales_agent.common.io import read_jsonl, write_jsonl
from sales_agent.common.schema import DialogueRecord, validate_dialogue

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "data" / "split.py"

ENERGY_KW = ["energy", "electricity", "kwh", "power bill", "tariff", "solar"]


def _m1_record(i: int, scenario: str = "general") -> dict:
    return {
        "id": f"dlg-m1-{i:04d}",
        "source": "hf:test",
        "scenario": scenario,
        "lang": "en",
        "n_turns": 1,
        "meta": {"raw_format": "prefixed_pairs"},
        "messages": [
            {"role": "user", "content": f"Distinct M1 user message number {i} about gadgets."},
            {"role": "assistant", "content": f"Distinct M1 assistant reply number {i} for you."},
        ],
    }


def _m2_record(i: int, scenario: str, n_turns: int = 2) -> dict:
    msgs = [{"role": "system", "content": "You are an ABC Energy sales agent."}]
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"M2 dialogue {i} user turn {t} on energy plans."})
        msgs.append(
            {"role": "assistant", "content": f"M2 dialogue {i} agent turn {t} reply distinctly."}
        )
    return {
        "id": f"dlg-m2-{i:04d}",
        "source": "synthetic:v1",
        "scenario": scenario,
        "lang": "en",
        "n_turns": n_turns,
        "meta": {"synth_model": "fake/model"},
        "messages": msgs,
    }


def build_inputs(workspace: Path, n_m2: int = 30) -> tuple[Path, Path]:
    """Tiny but split-able corpus: 81 M1 (mixed scenarios) + ``n_m2`` M2 (multi-turn)."""
    m1 = []
    for i in range(55):
        m1.append(_m1_record(i, "general"))
    for i in range(55, 70):
        m1.append(_m1_record(i, "objection_handling"))
    for i in range(70, 80):
        m1.append(_m1_record(i, "info_gathering"))
    # one keyword-only M1 record (general scenario, energy text)
    kw = _m1_record(999, "general")
    kw["messages"][0]["content"] = "My power bill and electricity tariff keep rising each month."
    m1.append(kw)

    m2 = []
    scenarios = ["objection_handling", "info_gathering", "cold_open", "closing", "general"]
    for i in range(n_m2):
        n = 2 if i % 2 == 0 else 4  # short (<=4) -- keeps strata simple
        m2.append(_m2_record(i, scenarios[i % len(scenarios)], n_turns=n))

    m1_path = workspace / "m1.jsonl"
    m2_path = workspace / "m2.jsonl"
    write_jsonl(m1_path, m1)
    write_jsonl(m2_path, m2)
    return m1_path, m2_path


def write_config(
    path: Path, output_dir: Path, m1: Path, m2: Path, *, ratio: float = 2.0, min_stratum: int = 5
) -> Path:
    cfg = {
        "seed": 42,
        "output_dir": str(output_dir),
        "smoke_limit": 10,
        "inputs": [
            {"path": str(m1), "role": "m1"},
            {"path": str(m2), "role": "m2"},
        ],
        "dedup": {"threshold": 0.85, "num_perm": 128, "shingle_size": 5},
        "downsample": {
            "ratio": ratio,
            "high_value_scenarios": ["objection_handling", "info_gathering"],
            "energy_keywords": ENERGY_KW,
            "energy_probe_baseline": 7,
        },
        "split": {"ratios": [0.9, 0.05, 0.05], "min_stratum": min_stratum},
        "dist_warn_pp": 3.0,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m3_split_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSplitCli:
    def test_full_run_writes_contract_outputs(self, tmp_path: Path):
        m1, m2 = build_inputs(tmp_path)
        out = tmp_path / "out"
        cfg = write_config(tmp_path / "cfg.yaml", out, m1, m2)
        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 0, proc.stderr

        # All three splits exist and every record is a valid DialogueRecord.
        for name in ("train", "val", "test"):
            recs = [DialogueRecord.model_validate(r) for r in read_jsonl(out / f"{name}.jsonl")]
            assert recs, f"{name} is empty"
            assert all(validate_dialogue(r) == [] for r in recs)

        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        # section 2.3 contract fields
        assert set(report["counts"]) == {"train", "val", "test"}
        assert report["ratios"] == [0.9, 0.05, 0.05]
        assert report["stratify_keys"] == ["scenario", "turn_bucket"]
        assert set(report["distribution"]) == {"train", "val", "test"}
        assert report["leakage_check"]["cross_split_dups"] == 0
        assert report["leakage_check"]["method"] == "minhash"
        # downsample wiring: M2 never downsampled; M1 capped to ratio * M2.
        ds = report["downsample"]
        assert ds["ratio"] == 2.0
        assert ds["m2_count"] == 30
        assert ds["target_m1"] == 60
        assert ds["final_m1"] == 60
        assert ds["m1_to_m2_ratio"] == 2.0
        assert ds["energy_keyword_deviation_vs_probe"] == ds["energy_keyword_hits"] - 7

        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["stats"]["counts"] == report["counts"]

    def test_no_id_across_splits(self, tmp_path: Path):
        m1, m2 = build_inputs(tmp_path)
        out = tmp_path / "out"
        cfg = write_config(tmp_path / "cfg.yaml", out, m1, m2)
        assert run_cli("--config", str(cfg)).returncode == 0
        ids = [
            r["id"]
            for name in ("train", "val", "test")
            for r in read_jsonl(out / f"{name}.jsonl")
        ]
        assert len(ids) == len(set(ids))

    def test_ratio_zero_is_m2_only(self, tmp_path: Path):
        # M2-only: need enough M2 to allocate val/test; pool into one stratum.
        m1, m2 = build_inputs(tmp_path, n_m2=80)
        out = tmp_path / "out"
        cfg = write_config(tmp_path / "cfg.yaml", out, m1, m2, ratio=0.0, min_stratum=40)
        assert run_cli("--config", str(cfg)).returncode == 0
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["downsample"]["final_m1"] == 0
        # every surviving record is synthetic (M2)
        for name in ("train", "val", "test"):
            for r in read_jsonl(out / f"{name}.jsonl"):
                assert r["source"].startswith("synthetic")

    def test_smoke_mode_caps_inputs(self, tmp_path: Path):
        m1, m2 = build_inputs(tmp_path)
        out = tmp_path / "out"
        # smoke_limit=60 (set in write_config below) keeps enough to split 90/5/5.
        cfg = write_config(tmp_path / "cfg.yaml", out, m1, m2, min_stratum=40)
        # bump smoke_limit for this case so val/test are non-empty.
        c = yaml.safe_load(cfg.read_text())
        c["smoke_limit"] = 60
        cfg.write_text(yaml.safe_dump(c))
        proc = run_cli("--config", str(cfg), "--smoke")
        assert proc.returncode == 0, proc.stderr
        total = sum(
            1 for name in ("train", "val", "test") for _ in read_jsonl(out / f"{name}.jsonl")
        )
        assert total < 81 + 30  # inputs were capped below the full corpus

    def test_missing_input_exits_2(self, tmp_path: Path):
        out = tmp_path / "out"
        cfg = write_config(
            tmp_path / "cfg.yaml", out, tmp_path / "nope1.jsonl", tmp_path / "nope2.jsonl"
        )
        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 2

    def test_invalid_record_exits_2(self, tmp_path: Path):
        # n_turns mismatch -> semantic validation failure -> contract exit.
        bad = _m1_record(0)
        bad["n_turns"] = 9
        m1_path = tmp_path / "m1.jsonl"
        write_jsonl(m1_path, [bad])
        m2_path = tmp_path / "m2.jsonl"
        write_jsonl(m2_path, [_m2_record(0, "general")])
        cfg = write_config(tmp_path / "cfg.yaml", tmp_path / "out", m1_path, m2_path)
        proc = run_cli("--config", str(cfg))
        assert proc.returncode == 2

    def test_detected_leak_forces_exit_2(self, tmp_path: Path, monkeypatch):
        """The leak->exit-2 wiring (a real leak can't occur post global dedup)."""
        from sales_agent.data.split import LeakageReport

        mod = _load_cli_module()
        m1, m2 = build_inputs(tmp_path)
        out = tmp_path / "out"
        cfg = write_config(tmp_path / "cfg.yaml", out, m1, m2)

        def fake_leak(*a, **k):
            return LeakageReport("minhash", 0.85, 3, [{"a": "x", "b": "y"}])

        monkeypatch.setattr(mod, "assert_no_leakage", fake_leak)
        rc = mod.main(["--config", str(cfg)])
        assert rc == 2
        # report is still written, recording the non-zero count.
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["leakage_check"]["cross_split_dups"] == 3
        # the splits themselves must NOT be written on a leak.
        assert not (out / "train.jsonl").exists()
