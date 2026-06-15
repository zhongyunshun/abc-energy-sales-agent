"""Unit tests for the four M9 rule metrics (T9.2), pos/neg fixtures for each.

Rules: made_up_price, over_length, role_break, no_question_in_gathering. Also the
deterministic token counter, RuleConfig.from_dict, and the apply_rules bundler.
The price patterns are hand-synced with M2 -- a guard test pins that.
"""

from __future__ import annotations

import pytest

from sales_agent.data.synthesize import DEFAULT_PRICE_PATTERNS as M2_PRICE_PATTERNS
from sales_agent.evals.rules import (
    DEFAULT_PRICE_PATTERNS,
    RULE_NAMES,
    RuleConfig,
    apply_rules,
    count_tokens,
    made_up_price,
    no_question_in_gathering,
    over_length,
    role_break,
)


@pytest.fixture
def cfg() -> RuleConfig:
    return RuleConfig.from_dict({"over_length_max_tokens": 10})


# --- price patterns kept in sync with M2 -----------------------------------


def test_price_patterns_synced_with_m2():
    # Module independence (no cross-import) but identical caliber: the literal
    # patterns must match M2's. If M2 changes, this fails so the copy is updated.
    assert DEFAULT_PRICE_PATTERNS == M2_PRICE_PATTERNS


# --- made_up_price ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Our rate is $0.09 per kWh.",
        "That comes to £50 a month.",
        "It would be 9.5 cents per kWh.",
        "About 12 p/kWh on the fixed plan.",
        "Roughly 35 dollars monthly.",
        "Around 30 pence per unit.",
    ],
)
def test_made_up_price_positive(text, cfg):
    assert made_up_price(text, cfg.price_patterns) is True


@pytest.mark.parametrize(
    "text",
    [
        "Could you tell me your monthly usage?",
        "Your bill shows 950 kWh, which helps.",  # bare usage, not a price
        "I can have the team prepare a personalized comparison.",
        "We have several plans depending on your usage.",
    ],
)
def test_made_up_price_negative(text, cfg):
    assert made_up_price(text, cfg.price_patterns) is False


# --- over_length -----------------------------------------------------------


def test_over_length_positive():
    assert over_length(121, 120) is True


def test_over_length_boundary_not_triggered():
    # Exactly at the threshold is NOT over.
    assert over_length(120, 120) is False


def test_over_length_negative():
    assert over_length(5, 120) is False


def test_count_tokens_words_and_punct():
    # 3 words + 1 period = 4 tokens.
    assert count_tokens("hello there friend.") == 4
    assert count_tokens("") == 0
    # Each punctuation mark counts; "?" is its own token.
    assert count_tokens("Why?") == 2


def test_over_length_via_count_tokens():
    short = "Thanks, that helps."
    long = " ".join(["word"] * 50)
    assert over_length(count_tokens(short), 10) is False
    assert over_length(count_tokens(long), 10) is True


# --- role_break ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "As an AI, I cannot make that promise.",
        "I am an AI assistant and cannot do that.",
        "As a large language model, I don't have feelings.",
        "I'm an AI, so I can't sign contracts.",
        "I am a language model trained by a company.",
    ],
)
def test_role_break_positive(text, cfg):
    assert role_break(text, cfg.role_break_patterns) is True


@pytest.mark.parametrize(
    "text",
    [
        "I'd be happy to help you switch providers.",
        "Let me check what options we can offer.",
        "As your energy advisor, I can walk you through the plans.",
    ],
)
def test_role_break_negative(text, cfg):
    assert role_break(text, cfg.role_break_patterns) is False


# --- no_question_in_gathering ----------------------------------------------


def test_no_question_in_gathering_positive(cfg):
    # info_gathering reply with no question -> flagged.
    assert no_question_in_gathering(
        "Thanks, I have noted your details.", "info_gathering", cfg.gathering_scenarios
    ) is True


def test_no_question_in_gathering_negative_has_question(cfg):
    assert no_question_in_gathering(
        "How much electricity do you use monthly?", "info_gathering", cfg.gathering_scenarios
    ) is False


def test_no_question_in_gathering_non_gathering_scenario_never_flags(cfg):
    # A statement in objection_handling must NOT trigger the gathering rule.
    assert no_question_in_gathering(
        "I completely understand your concern.", "objection_handling", cfg.gathering_scenarios
    ) is False


# --- RuleConfig.from_dict --------------------------------------------------


def test_ruleconfig_defaults():
    c = RuleConfig.from_dict({})
    assert c.over_length_max_tokens == 120
    assert c.gathering_scenarios == ("info_gathering",)
    assert len(c.price_patterns) == len(DEFAULT_PRICE_PATTERNS)
    assert len(c.role_break_patterns) > 0


def test_ruleconfig_none():
    c = RuleConfig.from_dict(None)
    assert c.over_length_max_tokens == 120


def test_ruleconfig_extra_price_patterns_appended():
    c = RuleConfig.from_dict({"extra_price_patterns": [r"\bquid\b"]})
    assert len(c.price_patterns) == len(DEFAULT_PRICE_PATTERNS) + 1
    assert made_up_price("about 40 quid", c.price_patterns) is True


def test_ruleconfig_role_patterns_replace():
    c = RuleConfig.from_dict({"role_break_patterns": [r"\brobot\b"]})
    assert len(c.role_break_patterns) == 1
    assert role_break("I am a robot", c.role_break_patterns) is True
    # The default "as an ai" pattern is gone (replaced).
    assert role_break("As an AI", c.role_break_patterns) is False


def test_ruleconfig_custom_gathering_scenarios():
    c = RuleConfig.from_dict({"gathering_scenarios": ["info_gathering", "qualification"]})
    assert no_question_in_gathering("No question here.", "qualification", c.gathering_scenarios)


# --- apply_rules -----------------------------------------------------------


def test_apply_rules_all_clean():
    cfg = RuleConfig.from_dict({"over_length_max_tokens": 120})
    flags, n = apply_rules(
        "Could you tell me your monthly usage?", "info_gathering", cfg
    )
    assert flags == {name: False for name in RULE_NAMES}
    assert n == count_tokens("Could you tell me your monthly usage?")


def test_apply_rules_multiple_violations():
    cfg = RuleConfig.from_dict({"over_length_max_tokens": 3})
    # Price + over_length + no question in gathering, in one short reply.
    flags, n = apply_rules("Our rate is $0.09 per kWh.", "info_gathering", cfg)
    assert flags["made_up_price"] is True
    assert flags["over_length"] is True
    assert flags["no_question_in_gathering"] is True
    assert flags["role_break"] is False


def test_apply_rules_returns_all_four_keys():
    cfg = RuleConfig.from_dict({})
    flags, _ = apply_rules("Hello there.", "general", cfg)
    assert set(flags) == set(RULE_NAMES)
