"""Unit tests for M2 synthesis logic (design doc section 3-M2, section 4).

All pure logic + the async orchestration against the programmable
``fake_openrouter`` -- zero real API calls.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import jinja2
import pytest

from sales_agent.common.io import read_jsonl
from sales_agent.common.openrouter import OpenRouterClient
from sales_agent.common.schema import DialogueRecord, PreferencePair, validate_dialogue
from sales_agent.data.synthesize import (
    SynthConfig,
    SynthError,
    SynthTask,
    build_prompt,
    contains_price,
    expand_task_matrix,
    extract_json,
    load_seeds,
    normalized_edit_distance,
    parse_and_validate,
    preference_id,
    run_synthesis,
    select_seeds,
)

REPO_ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SEEDS_DIR = REPO_ROOT / "configs" / "prompts" / "seeds"


class TransientError(Exception):
    """Retryable transient API failure used to drive the client's retry path."""


# ---------------------------------------------------------------------------
# Config / client helpers
# ---------------------------------------------------------------------------

DIALOGUE_SPEC = {
    "output_path": "data/interim/synthetic_dialogues.jsonl",
    "personas": ["persona_a", "persona_b"],
    "objection_types": ["none", "price"],
    "outcomes": ["agrees"],
    "n_turns_range": [3, 5],
    "scenarios": [
        {"name": "objection_handling", "quota": 5, "hint": "h", "directive": "d"},
        {"name": "info_gathering", "quota": 3},
    ],
}

PREFERENCE_SPEC = {
    "output_path": "data/interim/preference_pairs.jsonl",
    "personas": ["persona_a", "persona_b"],
    "context_scenarios": ["ctx_a", "ctx_b"],
    "failure_modes": [
        {"name": "pushy", "quota": 4, "desc": "x", "rejected_directive": "r"},
        {"name": "rate_hallucination", "quota": 2},
    ],
}


def make_cfg(**overrides) -> SynthConfig:
    cfg = {
        "seed": 42,
        "model": "fake/model",
        "concurrency": 8,
        "temperature": 0.9,
        "max_tokens": 1500,
        "max_retries": 2,
        "min_turns": 3,
        "min_edit_distance": 0.3,
        "n_seed_examples": 1,
        "dialogues": DIALOGUE_SPEC,
        "preferences": PREFERENCE_SPEC,
    }
    cfg.update(overrides)
    return SynthConfig.from_dict(cfg)


def make_client(fake, **kwargs) -> OpenRouterClient:
    kwargs.setdefault("backoff_base", 0)
    kwargs.setdefault("retryable", (TransientError,))
    return OpenRouterClient("fake/model", raw_client=fake, **kwargs)


TRIVIAL_TEMPLATE = jinja2.Environment(autoescape=False).from_string("{{ scenario }}")


def load_output_fixtures(name: str) -> list[dict]:
    return list(read_jsonl(FIXTURES / name))


def to_raw(entry: dict) -> str:
    """Reconstruct a raw model-output string from a fixture entry."""
    if "raw" in entry:
        return entry["raw"]
    s = json.dumps(entry["payload"])
    wrap = entry.get("wrap")
    if wrap == "fenced":
        return f"```json\n{s}\n```"
    if wrap == "prose":
        return f"Sure, here is the result:\n{s}\nHope that helps!"
    return s


# ---------------------------------------------------------------------------
# T2.2 expand_task_matrix
# ---------------------------------------------------------------------------


class TestExpandTaskMatrix:
    def test_dialogue_counts_match_quota(self):
        tasks = expand_task_matrix(make_cfg(), "dialogues")
        assert len(tasks) == 5 + 3
        per_scenario = {}
        for t in tasks:
            assert t.mode == "dialogues"
            per_scenario[t.scenario] = per_scenario.get(t.scenario, 0) + 1
        assert per_scenario == {"objection_handling": 5, "info_gathering": 3}

    def test_preference_counts_match_quota(self):
        tasks = expand_task_matrix(make_cfg(), "preferences")
        assert len(tasks) == 4 + 2
        per_mode = {}
        for t in tasks:
            assert t.mode == "preferences"
            per_mode[t.scenario] = per_mode.get(t.scenario, 0) + 1
        assert per_mode == {"pushy": 4, "rate_hallucination": 2}

    def test_prompt_vars_drawn_from_configured_dimensions(self):
        tasks = expand_task_matrix(make_cfg(), "dialogues")
        for t in tasks:
            assert t.prompt_vars["persona"] in DIALOGUE_SPEC["personas"]
            assert t.prompt_vars["objection_type"] in DIALOGUE_SPEC["objection_types"]
            assert t.prompt_vars["outcome"] in DIALOGUE_SPEC["outcomes"]
            assert 3 <= t.prompt_vars["n_turns"] <= 5

    def test_indices_are_unique_and_dense(self):
        tasks = expand_task_matrix(make_cfg(), "dialogues")
        assert [t.index for t in tasks] == list(range(len(tasks)))

    def test_seed_is_deterministic(self):
        a = expand_task_matrix(make_cfg(), "dialogues")
        b = expand_task_matrix(make_cfg(), "dialogues")
        assert [t.prompt_vars for t in a] == [t.prompt_vars for t in b]

    def test_different_seed_changes_sampling(self):
        a = expand_task_matrix(make_cfg(seed=1), "dialogues")
        b = expand_task_matrix(make_cfg(seed=999), "dialogues")
        assert [t.prompt_vars for t in a] != [t.prompt_vars for t in b]

    def test_per_scenario_limit_caps_each_quota(self):
        tasks = expand_task_matrix(make_cfg(), "dialogues", per_scenario_limit=2)
        per_scenario = {}
        for t in tasks:
            per_scenario[t.scenario] = per_scenario.get(t.scenario, 0) + 1
        assert per_scenario == {"objection_handling": 2, "info_gathering": 2}

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            expand_task_matrix(make_cfg(), "nonsense")

    def test_fixed_count_injects_one_example_per_task(self):
        seeds = {
            "objection_handling": [{"messages": ["seed0"]}, {"messages": ["seed1"]}],
            "info_gathering": [{"messages": ["ig0"]}],
        }
        tasks = expand_task_matrix(make_cfg(), "dialogues", seeds_by_key=seeds)
        oh = [t for t in tasks if t.scenario == "objection_handling"]
        assert all(len(t.prompt_vars["examples"]) == 1 for t in oh)  # n_seed_examples=1
        # random (seeded) selection draws on the whole pool across tasks
        seen = {ex for t in oh for ex in t.prompt_vars["examples"]}
        assert len(seen) == 2

    def test_range_count_stays_within_bounds(self):
        seeds = {
            "objection_handling": [{"messages": [i]} for i in range(4)],
            "info_gathering": [{"messages": [i]} for i in range(4)],
        }
        cfg = make_cfg(n_seed_examples=[1, 2])
        tasks = expand_task_matrix(cfg, "dialogues", seeds_by_key=seeds)
        assert all(1 <= len(t.prompt_vars["examples"]) <= 2 for t in tasks)
        # both counts should occur across enough tasks
        assert {len(t.prompt_vars["examples"]) for t in tasks} == {1, 2}

    def test_selection_is_deterministic_for_a_seed(self):
        seeds = {
            "objection_handling": [{"messages": [i]} for i in range(4)],
            "info_gathering": [{"messages": [i]} for i in range(4)],
        }
        a = expand_task_matrix(make_cfg(n_seed_examples=[1, 2]), "dialogues", seeds_by_key=seeds)
        b = expand_task_matrix(make_cfg(n_seed_examples=[1, 2]), "dialogues", seeds_by_key=seeds)
        assert [t.prompt_vars["examples"] for t in a] == [t.prompt_vars["examples"] for t in b]

    def test_no_seeds_means_empty_examples(self):
        tasks = expand_task_matrix(make_cfg(), "dialogues")
        assert all(t.prompt_vars["examples"] == [] for t in tasks)


class TestSelectSeeds:
    def test_samples_requested_count(self):
        pool = {"k": [{"a": 0}, {"a": 1}, {"a": 2}]}
        got = select_seeds(pool, "k", random.Random(0), 2)
        assert len(got) == 2 and all(g in pool["k"] for g in got)

    def test_count_greater_than_pool_is_clamped(self):
        pool = {"k": [{"a": 0}, {"a": 1}]}
        assert len(select_seeds(pool, "k", random.Random(0), 5)) == 2

    def test_missing_key_or_zero_count_returns_empty(self):
        pool = {"k": [{"a": 0}]}
        assert select_seeds(pool, "missing", random.Random(0), 1) == []
        assert select_seeds(pool, "k", random.Random(0), 0) == []

    def test_deterministic_for_same_rng_seed(self):
        pool = {"k": [{"a": i} for i in range(5)]}
        assert select_seeds(pool, "k", random.Random(7), 2) == select_seeds(
            pool, "k", random.Random(7), 2
        )


class TestLoadSeeds:
    def test_groups_by_key_field_and_strips_it(self):
        grouped = load_seeds(SEEDS_DIR / "dialogue_seeds.jsonl", "scenario")
        assert set(grouped) == {
            "objection_handling",
            "info_gathering",
            "cold_open",
            "closing",
            "general",
        }
        for examples in grouped.values():
            for ex in examples:
                assert "scenario" not in ex
                assert "messages" in ex

    def test_seed_dialogues_are_contract_valid_and_price_clean(self):
        """The seeds we feed the model must themselves pass our own gate."""
        grouped = load_seeds(SEEDS_DIR / "dialogue_seeds.jsonl", "scenario")
        cfg = make_cfg()
        for scenario, examples in grouped.items():
            for ex in examples:
                task = SynthTask("dialogues", scenario, 0, {})
                rec = parse_and_validate(json.dumps(ex), task, cfg)
                assert isinstance(rec, DialogueRecord), f"{scenario}: {rec}"
                assert validate_dialogue(rec) == []

    def test_seed_preferences_are_contract_valid(self):
        grouped = load_seeds(SEEDS_DIR / "preference_seeds.jsonl", "failure_mode")
        cfg = make_cfg()
        assert set(grouped) == {"pushy", "rate_hallucination"}
        for mode, examples in grouped.items():
            for ex in examples:
                task = SynthTask("preferences", mode, 0, {})
                pair = parse_and_validate(json.dumps(ex), task, cfg)
                assert isinstance(pair, PreferencePair), f"{mode}: {pair}"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_renders_system_plus_user_with_task_content(self):
        task = expand_task_matrix(make_cfg(), "dialogues")[0]
        msgs = build_prompt(task, TRIVIAL_TEMPLATE)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[1]["content"] == task.scenario

    def test_real_dialogue_template_includes_constraints_and_example(self):
        from sales_agent.data.synthesize import load_template

        seeds = load_seeds(SEEDS_DIR / "dialogue_seeds.jsonl", "scenario")
        task = expand_task_matrix(make_cfg(), "dialogues", seeds_by_key=seeds)[0]
        template = load_template(REPO_ROOT / "configs/prompts/synth_dialogue.j2")
        user = build_prompt(task, template)[1]["content"]
        assert "MUST NOT invent" in user
        assert task.prompt_vars["persona"] in user
        assert "Example 1:" in user  # seed example rendered


# ---------------------------------------------------------------------------
# T2.3 parse_and_validate -- dialogues
# ---------------------------------------------------------------------------


class TestParseDialogue:
    @pytest.mark.parametrize("entry", load_output_fixtures("synth_dialogue_outputs.jsonl"))
    def test_fixture_outputs(self, entry):
        cfg = make_cfg()
        task = SynthTask("dialogues", "objection_handling", 0, {"persona": "p"})
        result = parse_and_validate(to_raw(entry), task, cfg)
        if entry["ok"]:
            assert isinstance(result, DialogueRecord), result
            assert result.source == "synthetic:v1"
            assert result.scenario == "objection_handling"
            assert result.meta["synth_model"] == "fake/model"
            assert result.meta["template_version"] == "v1"
            assert validate_dialogue(result) == []
        else:
            assert isinstance(result, SynthError)
            assert result.kind == entry["kind"], (entry["label"], result)

    def test_meta_records_task_dimensions(self):
        cfg = make_cfg()
        good = next(
            e for e in load_output_fixtures("synth_dialogue_outputs.jsonl") if e["ok"]
        )
        task = SynthTask(
            "dialogues",
            "info_gathering",
            7,
            {"persona": "a retiree", "objection_type": "price", "outcome": "agrees"},
        )
        rec = parse_and_validate(to_raw(good), task, cfg)
        assert isinstance(rec, DialogueRecord)
        assert rec.meta["persona"] == "a retiree"
        assert rec.meta["objection_type"] == "price"
        assert rec.meta["outcome"] == "agrees"


# ---------------------------------------------------------------------------
# T2.3 parse_and_validate -- preferences
# ---------------------------------------------------------------------------


class TestParsePreference:
    @pytest.mark.parametrize("entry", load_output_fixtures("synth_preference_outputs.jsonl"))
    def test_fixture_outputs(self, entry):
        cfg = make_cfg()
        task = SynthTask("preferences", "pushy", 0, {"persona": "p"})
        result = parse_and_validate(to_raw(entry), task, cfg)
        if entry["ok"]:
            assert isinstance(result, PreferencePair), result
            assert result.scenario == "pushy"
            assert result.meta["synth_model"] == "fake/model"
        else:
            assert isinstance(result, SynthError)
            assert result.kind == entry["kind"], (entry["label"], result)

    def test_rejected_may_contain_price_but_chosen_may_not(self):
        cfg = make_cfg()
        task = SynthTask("preferences", "rate_hallucination", 0, {})
        payload = {
            "context": [{"role": "user", "content": "What's my rate?"}],
            "chosen": "It depends on your usage, so let me prepare an accurate quote.",
            "rejected": "It's a flat 9.5 cents per kWh, guaranteed for everyone.",
        }
        result = parse_and_validate(json.dumps(payload), task, cfg)
        assert isinstance(result, PreferencePair)  # price in rejected is allowed


# ---------------------------------------------------------------------------
# Price regex + edit distance + JSON extraction units
# ---------------------------------------------------------------------------


class TestContainsPrice:
    PATTERNS = make_cfg().price_patterns

    @pytest.mark.parametrize(
        "text",
        [
            "only $0.09 per kWh",
            "9.5 cents per kWh",
            "8.9 cents per kWh flat",
            "12 p/kWh",
            "save exactly 35 dollars a month",
            "£50 a month",
            "save $35 a month",
        ],
    )
    def test_flags_prices(self, text):
        assert contains_price(text, self.PATTERNS)

    @pytest.mark.parametrize(
        "text",
        [
            "your usage is about 950 kWh each month",
            "the price per kWh depends entirely on your usage",
            "the switch usually takes two to three weeks",
            "we serve a family of four with no problem",
            "I'm 100% sure we can help you compare plans",
            "let me prepare a personalized quote for you",
        ],
    )
    def test_ignores_non_prices(self, text):
        assert not contains_price(text, self.PATTERNS)


class TestEditDistance:
    def test_identical_is_zero(self):
        assert normalized_edit_distance("hello there", "hello there") == 0.0

    def test_case_and_space_insensitive(self):
        assert normalized_edit_distance("Hello  There", "hello there") == 0.0

    def test_completely_different_is_high(self):
        assert normalized_edit_distance("abcdef", "zyxwvu") == pytest.approx(1.0)

    def test_small_change_is_below_default_threshold(self):
        a = "Of course, I can help you compare your current plan."
        b = "Of course, I can help you compare your current plans."
        assert normalized_edit_distance(a, b) < 0.3


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_code_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped(self):
        assert extract_json('Sure!\n{"a": 1}\nDone.') == {"a": 1}

    def test_non_json_returns_none(self):
        assert extract_json("I cannot help with that.") is None

    def test_array_returns_none(self):
        assert extract_json("[1, 2, 3]") is None

    def test_empty_returns_none(self):
        assert extract_json("") is None


class TestPreferenceId:
    def test_stable_and_prefixed(self):
        from sales_agent.common.schema import Message

        ctx = [Message(role="user", content="hi")]
        a = preference_id(ctx, "good", "bad")
        b = preference_id(ctx, "good", "bad")
        assert a == b and a.startswith("pref-") and len(a) == len("pref-") + 12

    def test_differs_on_content(self):
        from sales_agent.common.schema import Message

        ctx = [Message(role="user", content="hi")]
        assert preference_id(ctx, "good", "bad") != preference_id(ctx, "good", "worse")


# ---------------------------------------------------------------------------
# T2.4 run_synthesis -- async orchestration with the fake client
# ---------------------------------------------------------------------------

GOOD_DIALOGUE = json.dumps(
    {
        "messages": [
            {"role": "system", "content": "You are an ABC Energy sales agent."},
            {"role": "user", "content": "Your rates seem high."},
            {"role": "assistant", "content": "I understand. May I ask what you pay now?"},
            {"role": "user", "content": "About a hundred a month."},
            {"role": "assistant", "content": "Thanks, I'll prepare a tailored comparison."},
            {"role": "user", "content": "Okay."},
            {"role": "assistant", "content": "Great, I'll send it over shortly."},
        ]
    }
)
BAD_DIALOGUE = "I'm sorry, I can't do that."


def single_task_cfg() -> SynthConfig:
    """A dialogue config that expands to exactly one task."""
    spec = dict(DIALOGUE_SPEC)
    spec["scenarios"] = [{"name": "objection_handling", "quota": 1}]
    spec["personas"] = ["p"]
    spec["objection_types"] = ["none"]
    spec["outcomes"] = ["agrees"]
    return make_cfg(dialogues=spec)


class TestRunSynthesis:
    async def test_all_succeed(self, fake_openrouter):
        cfg = make_cfg()
        tasks = expand_task_matrix(cfg, "dialogues")
        fake_openrouter.set_default(GOOD_DIALOGUE)
        client = make_client(fake_openrouter)
        result = await run_synthesis(tasks, client, cfg, TRIVIAL_TEMPLATE)
        assert result.attempted == len(tasks)
        assert result.succeeded == len(tasks)
        assert result.abandoned == 0
        assert all(r["source"] == "synthetic:v1" for r in result.records)
        assert result.usage.requests == len(tasks)

    async def test_retry_then_succeed(self, fake_openrouter):
        cfg = single_task_cfg()
        tasks = expand_task_matrix(cfg, "dialogues")
        assert len(tasks) == 1
        fake_openrouter.queue_responses(BAD_DIALOGUE, BAD_DIALOGUE, GOOD_DIALOGUE)
        client = make_client(fake_openrouter)
        result = await run_synthesis(tasks, client, cfg, TRIVIAL_TEMPLATE)
        assert result.succeeded == 1
        assert result.abandoned == 0
        assert result.validation_retries == 2  # two bad attempts before success
        assert fake_openrouter.call_count == 3

    async def test_abandon_after_max_retries(self, fake_openrouter):
        cfg = single_task_cfg()  # max_retries=2 -> 3 attempts total
        tasks = expand_task_matrix(cfg, "dialogues")
        fake_openrouter.set_default(BAD_DIALOGUE)
        client = make_client(fake_openrouter)
        result = await run_synthesis(tasks, client, cfg, TRIVIAL_TEMPLATE)
        assert result.succeeded == 0
        assert result.abandoned == 1
        assert result.errors_by_kind == {"not_json": 1}
        assert fake_openrouter.call_count == 3  # initial + 2 retries
        assert result.abandoned_samples[0]["kind"] == "not_json"

    async def test_api_error_counted_and_retried(self, fake_openrouter):
        cfg = single_task_cfg()
        tasks = expand_task_matrix(cfg, "dialogues")
        # client with no internal retries: each transient error surfaces as
        # OpenRouterError, which run_synthesis treats as an api_error and retries.
        for _ in range(3):
            fake_openrouter.queue_error(TransientError("down"))
        client = make_client(fake_openrouter, max_retries=0)
        result = await run_synthesis(tasks, client, cfg, TRIVIAL_TEMPLATE)
        assert result.succeeded == 0
        assert result.errors_by_kind == {"api_error": 1}
        assert fake_openrouter.call_count == 3

    async def test_mixed_outcomes_classify_errors(self, fake_openrouter):
        cfg = make_cfg()
        tasks = expand_task_matrix(cfg, "dialogues")  # 8 tasks
        # First task: price-in-assistant on every attempt; rest: good by default.
        priced = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are an ABC Energy agent."},
                    {"role": "user", "content": "What's my rate?"},
                    {"role": "assistant", "content": "Just 9 cents per kWh, the best around."},
                    {"role": "user", "content": "Hmm."},
                    {"role": "assistant", "content": "Shall I lock it in?"},
                    {"role": "user", "content": "No."},
                    {"role": "assistant", "content": "No problem."},
                ]
            }
        )
        for _ in range(3):  # 3 attempts for the first task before it is abandoned
            fake_openrouter.queue_response(priced)
        fake_openrouter.set_default(GOOD_DIALOGUE)
        client = make_client(fake_openrouter)
        result = await run_synthesis(tasks, client, cfg, TRIVIAL_TEMPLATE)
        assert result.errors_by_kind == {"price_in_assistant": 1}
        assert result.succeeded == len(tasks) - 1

    async def test_preferences_end_to_end(self, fake_openrouter):
        cfg = make_cfg()
        tasks = expand_task_matrix(cfg, "preferences")
        good_pref = json.dumps(
            {
                "context": [{"role": "user", "content": "Can you call back later?"}],
                "chosen": "Of course, I'll follow up at a time that suits you. When works best?",
                "rejected": "No, now is the only chance, sign up before this deal disappears!",
            }
        )
        fake_openrouter.set_default(good_pref)
        client = make_client(fake_openrouter)
        result = await run_synthesis(tasks, client, cfg, TRIVIAL_TEMPLATE)
        assert result.succeeded == len(tasks)
        assert all(r["id"].startswith("pref-") for r in result.records)
