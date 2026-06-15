"""Guard test for configs/eval_judge.yaml (design doc 3-M10 invariants).

The config is the single source of truth for the judge stage; this pins the
invariants so no magic number drifts into code: seed=42, judge models are
NON-Google (the whole point -- the synthesizer was google/gemini-2.5-flash, see
risk board row 39), n=100 per group, 1-5 score range, parse retries <= 2, and the
no-significant-difference threshold. Paths asserted OS-independently via
Path(...).as_posix().endswith(...). Pure file parsing, no API.
"""

from __future__ import annotations

from pathlib import Path

from sales_agent.common.config import load_config
from sales_agent.evals.judge import JudgeConfig

REPO = Path(__file__).resolve().parents[1]
JUDGE_CONFIG = REPO / "configs" / "eval_judge.yaml"


def _raw() -> dict:
    return load_config(JUDGE_CONFIG)


def posix(s: str) -> str:
    return Path(s).as_posix()


def test_seed_default():
    assert _raw()["seed"] == 42


def test_judge_models_non_google():
    # The core M10 requirement: judge must NOT be Google (gemini), to avoid
    # same-source bias with the gemini-flash synthesizer.
    models = _raw()["judge_models"]
    assert models, "judge_models must be non-empty"
    assert all(not m.lower().startswith("google/") for m in models), models
    assert all("gemini" not in m.lower() for m in models), models


def test_inputs_are_the_three_groups():
    inputs = [posix(p) for p in _raw()["inputs"]]
    assert any(p.endswith("reports/eval_offline/base") for p in inputs)
    assert any(p.endswith("reports/eval_offline/sft") for p in inputs)
    assert any(p.endswith("reports/eval_offline/dpo") for p in inputs)


def test_output_dir_wired():
    assert posix(_raw()["output_dir"]).endswith("reports/eval_judge")


def test_prompt_template_wired():
    assert posix(_raw()["prompt_template_path"]).endswith("configs/prompts/judge.j2")


def test_sampling_default_100():
    assert _raw()["sampling"]["n_samples"] == 100


def test_smoke_block():
    assert _raw()["smoke"]["n_samples"] >= 1


def test_score_range_and_threshold():
    raw = _raw()
    assert raw["score_min"] == 1 and raw["score_max"] == 5
    assert raw["no_diff_threshold"] == 0.3


def test_parse_retries_within_design_bound():
    # Design doc 3-M10: re-ask a malformed judge response at most twice.
    assert _raw()["max_retries"] <= 2


def test_reproducible_judging():
    raw = _raw()
    assert raw["temperature"] == 0.0  # deterministic scoring
    assert isinstance(raw["max_tokens"], int) and raw["max_tokens"] > 0


def test_from_dict_parses():
    # The config must load into the typed JudgeConfig the CLI relies on.
    cfg = JudgeConfig.from_dict(_raw())
    assert cfg.seed == 42
    assert len(cfg.judge_models) >= 1
    assert cfg.dimensions == ("coherence", "sales_logic", "professionalism", "hallucination")
