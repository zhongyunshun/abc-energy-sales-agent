"""End-to-end tests for the M9 thin CLI (scripts/eval/run_offline_eval.py).

Runs against a tiny test.jsonl with generation monkeypatched in-process (no
endpoint, no GPU): verifies the contract artifacts (results.jsonl rows with
rule_flags + summary.json + manifest.json), that the reply is reasoning-stripped
before scoring, and the contract-failure exit code.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from sales_agent.common.io import read_jsonl, write_jsonl
from sales_agent.evals.generate import GenOutput

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "eval" / "run_offline_eval.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m9_eval_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dialogue(i: int, scenario: str) -> dict:
    return {
        "id": f"dlg-{i:04d}",
        "source": "synthetic:v1",
        "scenario": scenario,
        "lang": "en",
        "n_turns": 1,
        "meta": {},
        "messages": [
            {"role": "user", "content": f"Question {i} about my energy plan."},
            {"role": "assistant", "content": f"Reference reply {i}."},
        ],
    }


def _write_config(workspace: Path, test_path: Path, out_dir: Path) -> Path:
    cfg = {
        "seed": 42,
        "test_path": str(test_path),
        "output_dir": str(out_dir),
        "generation": {"temperature": 0.0, "max_tokens": 256, "concurrency": 4},
        "sampling": {"n_samples": None},
        "rules": {"over_length_max_tokens": 5, "gathering_scenarios": ["info_gathering"]},
        "smoke": {"n_samples": 2},
        "endpoint": {"default": "http://127.0.0.1:8000/v1", "served_model": "sales-agent-awq"},
    }
    path = workspace / "eval_offline.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_full_run_writes_contract_outputs(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    write_jsonl(test_path, [
        _dialogue(0, "info_gathering"),
        _dialogue(1, "objection_handling"),
        _dialogue(2, "general"),
    ])
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, test_path, out_dir)

    mod = _load_cli_module()

    # Canned replies (one per sample, by order): a clean question, an over-length
    # reply with a price + a leading empty <think> (must be stripped), and a short one.
    replies = [
        "How much electricity do you use each month?",
        "<think>\n\n</think>\n\nOur rate is $0.09 per kWh and this is a very long reply indeed.",
        "Thanks!",
    ]

    async def fake_generate_all(batch, client, model, gen_cfg):
        return [GenOutput(content=replies[i], usage_completion_tokens=10)
                for i in range(len(batch))]

    monkeypatch.setattr(mod, "generate_all", fake_generate_all)

    rc = mod.main(["--config", str(cfg), "--model-tag", "dpo"])
    assert rc == 0

    tag_dir = out_dir / "dpo"
    rows = list(read_jsonl(tag_dir / "results.jsonl"))
    assert len(rows) == 3
    by_id = {r["id"]: r for r in rows}

    # Reasoning prefix stripped before scoring + storing.
    r1 = by_id["dlg-0001"]
    assert "<think>" not in r1["completion"]
    assert r1["completion"].startswith("Our rate is $0.09")
    assert r1["rule_flags"]["made_up_price"] is True
    assert r1["rule_flags"]["over_length"] is True  # > 5 token budget
    assert "rule_flags" in r1 and "gen_config" in r1 and "n_tokens" in r1

    # info_gathering reply DOES ask a question -> not flagged.
    assert by_id["dlg-0000"]["rule_flags"]["no_question_in_gathering"] is False

    summary = json.loads((tag_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_tag"] == "dpo"
    assert summary["n_samples"] == 3
    assert summary["gen_config"] == {"temperature": 0.0, "max_tokens": 256}
    assert summary["overall"]["rule_counts"]["made_up_price"] == 1
    assert "info_gathering" in summary["by_scenario"]

    assert (tag_dir / "manifest.json").exists()
    manifest = json.loads((tag_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["model_tag"] == "dpo"
    assert manifest["stats"]["n_samples"] == 3


def test_smoke_caps_samples(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    write_jsonl(test_path, [_dialogue(i, "general") for i in range(10)])
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, test_path, out_dir)
    mod = _load_cli_module()

    async def fake_generate_all(batch, client, model, gen_cfg):
        assert len(batch) == 2  # smoke.n_samples
        return [GenOutput(content="hi", usage_completion_tokens=1) for _ in batch]

    monkeypatch.setattr(mod, "generate_all", fake_generate_all)
    rc = mod.main(["--config", str(cfg), "--model-tag", "base", "--smoke"])
    assert rc == 0
    assert len(list(read_jsonl(out_dir / "base" / "results.jsonl"))) == 2


def test_missing_test_set_exit_2(tmp_path: Path):
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, tmp_path / "does_not_exist.jsonl", out_dir)
    mod = _load_cli_module()
    rc = mod.main(["--config", str(cfg), "--model-tag", "dpo"])
    assert rc == 2
