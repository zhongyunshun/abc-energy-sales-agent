"""Behavioral rule metrics for offline eval (design doc 3-M9 step 3, proposal 4-D1).

Pure logic only (no torch / transformers / network) -- unit-tested on the host.
Each rule maps a (already reasoning-stripped) assistant reply to a boolean flag;
:func:`apply_rules` bundles the four flags plus the reply's token count for the
``rule_flags`` field of the M9 results contract (design doc 2.4).

The four rules (proposal 4-D1: a voice sales agent must not invent prices, must
stay in role and compliant, and must keep replies short):

- ``made_up_price``: the reply contains a concrete price/rate figure -> a
  hallucinated quote. The patterns are kept in sync *by hand* with M2's
  ``data/synthesize.py::DEFAULT_PRICE_PATTERNS`` (same caliber as the generation
  gate) -- NOT imported, because modules interact only through file contracts
  (design doc 1.1); see :data:`DEFAULT_PRICE_PATTERNS` below.
- ``over_length``: token count > threshold (default 120) -- too long for speech.
- ``role_break``: an "as an AI / language model" disclosure that breaks the
  sales-agent persona.
- ``no_question_in_gathering``: in an info-gathering turn the reply asks no
  question (a gathering agent should probe for usage / bill / decision-maker).

Token counting (over_length) uses a deterministic, dependency-free word/punct
counter so the length metric is IDENTICAL across all three model groups and runs
on the host without a tokenizer (autonomous decision, confirmed: numbers go into
the README). It approximates model tokens; the endpoint's true
``usage.completion_tokens`` is recorded separately as a reference field by the CLI.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Same caliber as data/synthesize.py::DEFAULT_PRICE_PATTERNS (M2 generation gate).
# Hand-synced copy -- the modules share a contract, not code (design doc 1.1). If
# you change one, change the other. Matches currency-prefixed amounts and per-unit
# energy rates while leaving bare usage figures ("950 kWh") and word numbers alone.
DEFAULT_PRICE_PATTERNS: tuple[str, ...] = (
    r"[$£€]\s?\d",                                                  # $0.09, £50, € 30
    r"\d+(?:\.\d+)?\s*(?:cents?|pence|p)\s*(?:/|per)\s*kwh",        # 9.5 cents per kWh, 12 p/kWh
    r"\d+(?:\.\d+)?\s*(?:cents?|pence)\b",                          # 9.5 cents
    r"\d+(?:\.\d+)?\s*(?:dollars?|pounds?|euros?|usd|gbp|eur)\b",   # 35 dollars
)

# "Out of character" disclosures that break the sales-agent persona.
DEFAULT_ROLE_BREAK_PATTERNS: tuple[str, ...] = (
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bas a large language model\b",
    r"\bi am an ai\b",
    r"\bi'?m an ai\b",
    r"\bi am a language model\b",
    r"\blanguage model\b",
    r"\bai language model\b",
)

DEFAULT_GATHERING_SCENARIOS: tuple[str, ...] = ("info_gathering",)
DEFAULT_OVER_LENGTH_MAX_TOKENS = 120

# Words and standalone punctuation marks. Deterministic and host-only; see module
# docstring on why this (not a real tokenizer) is the canonical length metric.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

RULE_NAMES: tuple[str, ...] = (
    "made_up_price",
    "over_length",
    "role_break",
    "no_question_in_gathering",
)


def count_tokens(text: str) -> int:
    """Deterministic word + punctuation token count (approximates model tokens)."""
    return len(_TOKEN_RE.findall(text))


def made_up_price(text: str, patterns: Iterable[re.Pattern]) -> bool:
    """True if the reply states a concrete price/rate (hallucinated quote)."""
    return any(p.search(text) for p in patterns)


def over_length(n_tokens: int, max_tokens: int) -> bool:
    """True if the reply exceeds the length budget (too long for speech)."""
    return n_tokens > max_tokens


def role_break(text: str, patterns: Iterable[re.Pattern]) -> bool:
    """True if the reply breaks character with an AI/language-model disclosure."""
    return any(p.search(text) for p in patterns)


def no_question_in_gathering(text: str, scenario: str, gathering_scenarios: Sequence[str]) -> bool:
    """True only for an info-gathering turn whose reply contains no question mark.

    Non-gathering scenarios never trigger (always False); within a gathering
    scenario the rule fires when the reply asks nothing ("?" absent).
    """
    if scenario not in gathering_scenarios:
        return False
    return "?" not in text


def _compile(patterns: Iterable[str]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


@dataclass(frozen=True)
class RuleConfig:
    """Thresholds and compiled patterns for the four rules (from eval_offline.yaml)."""

    over_length_max_tokens: int = DEFAULT_OVER_LENGTH_MAX_TOKENS
    gathering_scenarios: tuple[str, ...] = DEFAULT_GATHERING_SCENARIOS
    price_patterns: tuple[re.Pattern, ...] = ()
    role_break_patterns: tuple[re.Pattern, ...] = ()

    @classmethod
    def from_dict(cls, cfg: dict | None) -> RuleConfig:
        """Build from the ``rules`` block; defaults fill anything omitted.

        ``extra_price_patterns`` are appended to the default price patterns;
        ``role_break_patterns`` (when non-empty) REPLACE the defaults.
        """
        cfg = cfg or {}
        price = list(_compile(DEFAULT_PRICE_PATTERNS))
        price += list(_compile(cfg.get("extra_price_patterns") or ()))
        role_src = cfg.get("role_break_patterns") or DEFAULT_ROLE_BREAK_PATTERNS
        gathering = tuple(cfg.get("gathering_scenarios") or DEFAULT_GATHERING_SCENARIOS)
        return cls(
            over_length_max_tokens=cfg.get(
                "over_length_max_tokens", DEFAULT_OVER_LENGTH_MAX_TOKENS
            ),
            gathering_scenarios=gathering,
            price_patterns=tuple(price),
            role_break_patterns=_compile(role_src),
        )


def apply_rules(completion: str, scenario: str, cfg: RuleConfig) -> tuple[dict[str, bool], int]:
    """Evaluate all four rules on a clean (reasoning-stripped) reply.

    Returns ``(rule_flags, n_tokens)``. ``completion`` MUST already be passed
    through :func:`samples.strip_reasoning` so an empty/leading ``<think>`` block
    cannot inflate the token count or trip role_break (risk board Option A).
    """
    n_tokens = count_tokens(completion)
    flags = {
        "made_up_price": made_up_price(completion, cfg.price_patterns),
        "over_length": over_length(n_tokens, cfg.over_length_max_tokens),
        "role_break": role_break(completion, cfg.role_break_patterns),
        "no_question_in_gathering": no_question_in_gathering(
            completion, scenario, cfg.gathering_scenarios
        ),
    }
    return flags, n_tokens
