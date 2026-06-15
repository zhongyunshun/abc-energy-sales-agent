"""Tests for the M2 thin CLI (scripts/data/synthesize.py).

Runs ``main()`` in-process with an injected fake-backed OpenRouterClient, so
there are zero real API calls. The missing-key path is exercised without
constructing a real client.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sales_agent.common.openrouter import API_KEY_ENV_VAR, OpenRouterClient

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "data" / "synthesize.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("m2_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cli = _load_cli_module()

GOOD_DIALOGUE = json.dumps(
    {
        "messages": [
            {"role": "system", "content": "You are an ABC Energy sales agent."},
            {"role": "user", "content": "Your rates seem high."},
            {"role": "assistant", "content": "I understand. What do you pay now?"},
            {"role": "user", "content": "About a hundred a month."},
            {"role": "assistant", "content": "Thanks, I'll prepare a tailored comparison."},
            {"role": "user", "content": "Okay."},
            {"role": "assistant", "content": "Great, I'll send it over."},
        ]
    }
)
GOOD_PREFERENCE = json.dumps(
    {
        "context": [{"role": "user", "content": "Can you call back later?"}],
        "chosen": "Of course, I'll follow up when it suits you. When is best?",
        "rejected": "No, now is the only chance, sign up before this deal disappears!",
    }
)


def write_config(path: Path, output_dir: Path) -> Path:
    """A small but complete synthesize config pointing outputs into a tmp dir."""
    cfg = {
        "seed": 42,
        "model": "fake/model",
        "concurrency": 4,
        "max_retries": 1,
        "min_turns": 3,
        "min_edit_distance": 0.3,
        "n_seed_examples": 1,
        "smoke_per_scenario": 2,
        "pricing": {"input_per_1m": 0.3, "output_per_1m": 2.5},
        "templates": {
            "dialogue_path": str(REPO_ROOT / "configs/prompts/synth_dialogue.j2"),
            "preference_path": str(REPO_ROOT / "configs/prompts/synth_preference.j2"),
        },
        "seeds": {
            "dialogue_path": str(REPO_ROOT / "configs/prompts/seeds/dialogue_seeds.jsonl"),
            "preference_path": str(REPO_ROOT / "configs/prompts/seeds/preference_seeds.jsonl"),
        },
        "dialogues": {
            "output_path": str(output_dir / "synthetic_dialogues.jsonl"),
            "n_turns_range": [3, 5],
            "personas": ["persona_a", "persona_b"],
            "objection_types": ["none", "price"],
            "outcomes": ["agrees"],
            "scenarios": [
                {"name": "objection_handling", "quota": 3},
                {"name": "info_gathering", "quota": 2},
            ],
        },
        "preferences": {
            "output_path": str(output_dir / "preference_pairs.jsonl"),
            "personas": ["persona_a"],
            "context_scenarios": ["ctx_a"],
            "failure_modes": [
                {"name": "pushy", "quota": 2},
                {"name": "rate_hallucination", "quota": 2},
            ],
        },
    }
    import yaml

    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def fake_client(fake, default: str) -> OpenRouterClient:
    fake.set_default(default)
    return OpenRouterClient("fake/model", raw_client=fake, backoff_base=0)


class TestSynthesizeCli:
    def test_dialogues_full_run_writes_outputs(self, tmp_path, fake_openrouter):
        out = tmp_path / "interim"
        cfg = write_config(tmp_path / "cfg.yaml", out)
        rc = cli.main(
            ["--config", str(cfg), "--mode", "dialogues"],
            client=fake_client(fake_openrouter, GOOD_DIALOGUE),
        )
        assert rc == 0

        records = list(_read_jsonl(out / "synthetic_dialogues.jsonl"))
        assert len(records) == 5  # 3 + 2 quota
        assert all(r["source"] == "synthetic:v1" for r in records)
        assert {r["scenario"] for r in records} == {"objection_handling", "info_gathering"}
        assert all(r["meta"]["synth_model"] == "fake/model" for r in records)

        report = json.loads((out / "dialogues_cost_report.json").read_text())
        assert report["succeeded"] == 5
        assert report["usage"]["requests"] == 5
        assert report["estimated_cost_usd"] is not None

        manifest = json.loads((out / "dialogues_manifest.json").read_text())
        assert manifest["stats"]["succeeded"] == 5
        assert len(manifest["inputs"]) == 2  # template + seed file hashed

    def test_preferences_run_writes_pref_ids(self, tmp_path, fake_openrouter):
        out = tmp_path / "interim"
        cfg = write_config(tmp_path / "cfg.yaml", out)
        rc = cli.main(
            ["--config", str(cfg), "--mode", "preferences"],
            client=fake_client(fake_openrouter, GOOD_PREFERENCE),
        )
        assert rc == 0
        records = list(_read_jsonl(out / "preference_pairs.jsonl"))
        assert len(records) == 4
        assert all(r["id"].startswith("pref-") for r in records)
        assert {r["scenario"] for r in records} == {"pushy", "rate_hallucination"}

    def test_smoke_caps_per_scenario(self, tmp_path, fake_openrouter):
        out = tmp_path / "interim"
        cfg = write_config(tmp_path / "cfg.yaml", out)
        rc = cli.main(
            ["--config", str(cfg), "--mode", "dialogues", "--smoke"],
            client=fake_client(fake_openrouter, GOOD_DIALOGUE),
        )
        assert rc == 0
        records = list(_read_jsonl(out / "synthetic_dialogues.jsonl"))
        assert len(records) == 4  # smoke_per_scenario=2 across 2 scenarios

    def test_output_dir_override(self, tmp_path, fake_openrouter):
        out = tmp_path / "interim"
        override = tmp_path / "elsewhere"
        cfg = write_config(tmp_path / "cfg.yaml", out)
        rc = cli.main(
            ["--config", str(cfg), "--mode", "dialogues", "--output-dir", str(override)],
            client=fake_client(fake_openrouter, GOOD_DIALOGUE),
        )
        assert rc == 0
        assert (override / "synthetic_dialogues.jsonl").exists()
        assert not (out / "synthetic_dialogues.jsonl").exists()

    def test_all_abandoned_exits_contract(self, tmp_path, fake_openrouter):
        out = tmp_path / "interim"
        cfg = write_config(tmp_path / "cfg.yaml", out)
        rc = cli.main(
            ["--config", str(cfg), "--mode", "dialogues"],
            client=fake_client(fake_openrouter, "not json at all"),
        )
        assert rc == cli.EXIT_CONTRACT
        # cost report is still written for diagnosis
        report = json.loads((out / "dialogues_cost_report.json").read_text())
        assert report["succeeded"] == 0
        assert report["errors_by_kind"]["not_json"] > 0

    def test_missing_api_key_exits_external(self, tmp_path, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        # Point the env loader at a non-existent file so the real key is unseen.
        monkeypatch.setattr(cli, "find_repo_root", lambda *a, **k: tmp_path)
        out = tmp_path / "interim"
        cfg = write_config(tmp_path / "cfg.yaml", out)
        rc = cli.main(["--config", str(cfg), "--mode", "dialogues"])  # client=None
        assert rc == cli.EXIT_EXTERNAL


def _read_jsonl(path: Path):
    from sales_agent.common.io import read_jsonl

    return read_jsonl(path)


def test_load_env_file_sets_missing_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('FOO_KEY="abc123"\n# comment\nBAR=plain\n', encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    cli.load_env_file(env)
    import os

    assert os.environ["FOO_KEY"] == "abc123"
    assert os.environ["BAR"] == "plain"


def test_load_env_file_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "fromenv")
    cli.load_env_file(env)
    import os

    assert os.environ["FOO_KEY"] == "fromenv"


@pytest.mark.parametrize("mode", ["dialogues", "preferences"])
def test_real_config_loads_and_expands(mode):
    """The committed configs/synthesize.yaml must expand without error."""
    from sales_agent.common.config import load_config
    from sales_agent.data.synthesize import SynthConfig, expand_task_matrix

    cfg_dict = load_config(REPO_ROOT / "configs" / "synthesize.yaml")
    cfg = SynthConfig.from_dict(cfg_dict)
    tasks = expand_task_matrix(cfg, mode, per_scenario_limit=2)
    assert tasks and all(t.mode == mode for t in tasks)
