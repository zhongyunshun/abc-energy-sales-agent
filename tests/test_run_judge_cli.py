"""End-to-end tests for the M10 thin CLI (scripts/eval/run_judge.py).

Runs against tiny per-group results.jsonl with the OpenRouter client replaced by
the conftest fake (zero real API calls): verifies the contract artifacts
(scores.jsonl + comparison.md + aggregate.json + manifest.json), the same-id batch
across groups, --smoke capping, and the contract / dependency exit codes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from sales_agent.common.io import read_jsonl, write_jsonl
from sales_agent.common.openrouter import OpenRouterClient

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "eval" / "run_judge.py"
TEMPLATE = REPO_ROOT / "configs" / "prompts" / "judge.j2"

GOOD = json.dumps(
    {
        "coherence": {"score": 4, "reason": "ok"},
        "sales_logic": {"score": 3, "reason": "ok"},
        "professionalism": {"score": 5, "reason": "ok"},
        "hallucination": {"score": 5, "reason": "ok"},
    }
)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m10_judge_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(i: int, scenario: str, tag: str) -> dict:
    return {
        "id": f"dlg-{i:04d}",
        "scenario": scenario,
        "prompt_messages": [
            {"role": "system", "content": "You are an ABC Energy agent."},
            {"role": "user", "content": f"Customer turn {i}."},
        ],
        "completion": f"{tag} reply {i}",
        "rule_flags": {},
        "n_tokens": 5,
    }


def _make_inputs(tmp_path: Path, n: int = 8) -> list[Path]:
    scen = ["general"] * (n - 2) + ["objection_handling", "info_gathering"]
    dirs = []
    for tag in ("base", "sft", "dpo"):
        d = tmp_path / "eval_offline" / tag
        write_jsonl(d / "results.jsonl", [_row(i, scen[i], tag) for i in range(n)])
        dirs.append(d)
    return dirs


def _write_config(tmp_path: Path, out_dir: Path, *, n: int = 4, judges=None) -> Path:
    cfg = {
        "seed": 42,
        "output_dir": str(out_dir),
        "prompt_template_path": str(TEMPLATE),
        "judge_models": judges or ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4"],
        "sampling": {"n_samples": n},
        "temperature": 0.0,
        "max_tokens": 64,
        "max_retries": 2,
        "client_max_retries": 0,
        "concurrency": 4,
        "use_json_mode": False,
        "dimensions": ["coherence", "sales_logic", "professionalism", "hallucination"],
        "score_min": 1,
        "score_max": 5,
        "no_diff_threshold": 0.3,
        "smoke": {"n_samples": 5},
        "pricing": {"anthropic/claude-sonnet-4.6": {"input_per_m": 3.0, "output_per_m": 15.0}},
    }
    path = tmp_path / "eval_judge.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _patch_client(mod, fake):
    mod._build_client = lambda cfg: OpenRouterClient(
        model=cfg.judge_models[0], raw_client=fake, backoff_base=0, max_retries=0
    )


def test_full_run_writes_contract_outputs(tmp_path: Path, fake_openrouter):
    dirs = _make_inputs(tmp_path, n=8)
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir, n=4)
    fake_openrouter.set_default(GOOD)

    mod = _load_cli_module()
    _patch_client(mod, fake_openrouter)

    rc = mod.main(["--config", str(cfg), "--inputs", *[str(d) for d in dirs]])
    assert rc == 0

    scores = list(read_jsonl(out_dir / "scores.jsonl"))
    # 2 judges x 3 tags x 4 samples = 24 score rows.
    assert len(scores) == 24
    s0 = scores[0]
    assert set(s0["scores"]) == {"coherence", "sales_logic", "professionalism", "hallucination"}
    assert "judge_model" in s0 and "rationale" in s0 and "judge_raw" in s0

    # The SAME 4 ids judged for every group (the DoD guarantee).
    ids_by_tag = {}
    for r in scores:
        ids_by_tag.setdefault(r["model_tag"], set()).add(r["id"])
    assert ids_by_tag["base"] == ids_by_tag["sft"] == ids_by_tag["dpo"]
    assert len(ids_by_tag["base"]) == 4

    assert (out_dir / "comparison.md").exists()
    md = (out_dir / "comparison.md").read_text(encoding="utf-8")
    assert "LLM-as-a-Judge" in md and "coherence" in md

    agg = json.loads((out_dir / "aggregate.json").read_text(encoding="utf-8"))
    assert set(agg["judges"]) == {"anthropic/claude-sonnet-4.6", "openai/gpt-5.4"}
    assert agg["model_tags"] == ["base", "sft", "dpo"]

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["n_per_group"] == 4
    assert manifest["stats"]["succeeded"] == 24
    # Cost: sonnet priced, gpt-5.4 has no pricing entry -> flagged, not silently 0.
    assert "openai/gpt-5.4" in manifest["stats"]["cost"]["missing_pricing"]


def test_smoke_caps_samples(tmp_path: Path, fake_openrouter):
    dirs = _make_inputs(tmp_path, n=8)
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir, judges=["judgeA"])
    fake_openrouter.set_default(GOOD)
    mod = _load_cli_module()
    _patch_client(mod, fake_openrouter)

    rc = mod.main(["--config", str(cfg), "--inputs", *[str(d) for d in dirs], "--smoke"])
    assert rc == 0
    scores = list(read_jsonl(out_dir / "scores.jsonl"))
    # 1 judge x 3 tags x 5 (smoke) = 15.
    assert len(scores) == 15
    per_group = {}
    for r in scores:
        per_group.setdefault(r["model_tag"], 0)
        per_group[r["model_tag"]] += 1
    assert per_group == {"base": 5, "sft": 5, "dpo": 5}


def test_missing_input_dir_exit_2(tmp_path: Path):
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir)
    mod = _load_cli_module()
    rc = mod.main(["--config", str(cfg), "--inputs", str(tmp_path / "nope")])
    assert rc == 2


def test_all_judge_calls_fail_exit_3(tmp_path: Path, fake_openrouter):
    dirs = _make_inputs(tmp_path, n=8)
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir, n=2, judges=["judgeA"])
    fake_openrouter.set_default("not json at all")  # every call unparseable
    mod = _load_cli_module()
    _patch_client(mod, fake_openrouter)
    rc = mod.main(["--config", str(cfg), "--inputs", *[str(d) for d in dirs]])
    assert rc == 3
