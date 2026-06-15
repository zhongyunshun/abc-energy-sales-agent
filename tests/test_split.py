"""Unit tests for M3 split & leakage logic (src/sales_agent/data/split.py).

All pure CPU, no GPU and no API. Covers: MinHash near-dedup (recall + keep
priority + idempotence), M1 downsampling (high-value always kept, ratio
variants, high-value-exceeds-target edge, idempotence), stratified split
(ratios, distribution, small-stratum merge, whole-dialogue, idempotence), and
the cross-split leakage assertion (injected leak must be detected).
"""

from __future__ import annotations

from sales_agent.common.schema import DialogueRecord, Message
from sales_agent.data.split import (
    assert_no_leakage,
    compile_energy_matcher,
    default_stratum_key,
    distribution_warnings,
    downsample_m1,
    is_real,
    minhash_dedup,
    scenario_distribution,
    stratified_split,
    turn_bucket,
)


def make_record(
    rec_id: str,
    *turns: tuple[str, str],
    scenario: str = "general",
    source: str = "hf:test",
    n_turns: int | None = None,
) -> DialogueRecord:
    msgs = [Message(role=r, content=c) for r, c in turns]
    nt = n_turns if n_turns is not None else sum(1 for m in msgs if m.role == "assistant")
    return DialogueRecord(
        id=rec_id,
        source=source,
        scenario=scenario,
        lang="en",
        n_turns=nt,
        messages=msgs,
    )


def single_turn(rec_id: str, user: str, assistant: str = "Sure, happy to help.", **kw):
    return make_record(rec_id, ("user", user), ("assistant", assistant), **kw)


# ---------------------------------------------------------------------------
# Helpers / small mappers
# ---------------------------------------------------------------------------


class TestTurnBucket:
    def test_boundaries(self):
        assert turn_bucket(1) == "short"
        assert turn_bucket(4) == "short"
        assert turn_bucket(5) == "mid"
        assert turn_bucket(8) == "mid"
        assert turn_bucket(9) == "long"

    def test_default_stratum_key(self):
        rec = make_record("dlg-x", ("user", "hi"), ("assistant", "yo"), scenario="closing")
        assert default_stratum_key(rec) == ("closing", "short")

    def test_is_real(self):
        assert is_real(make_record("a", ("user", "x"), ("assistant", "y"), source="hf:ds"))
        assert not is_real(
            make_record("b", ("user", "x"), ("assistant", "y"), source="synthetic:v1")
        )


# ---------------------------------------------------------------------------
# T3.1 -- minhash_dedup
# ---------------------------------------------------------------------------


class TestMinhashDedup:
    def test_empty(self):
        kept, stats = minhash_dedup([], threshold=0.85, seed=42)
        assert kept == []
        assert stats.n_input == 0 and stats.n_kept == 0 and stats.n_dropped == 0

    # Long base (~76 normalized tokens) so a small tail append keeps the MinHash
    # *estimated* Jaccard well above 0.85 despite 128-perm estimation variance.
    # (Mid-text inserts and short bases are unreliable at this perm count.)
    LU = (
        "I am reaching out from ABC Energy regarding your current electricity plan and "
        "whether you might save on your monthly power bill by switching supplier today, "
        "because many local households have lowered their winter heating costs already "
        "this year with our fixed tariff."
    )
    AS = (
        "Thank you for explaining, that does sound reasonable and I would be open to "
        "reviewing my current electricity usage and monthly bill to see if switching "
        "could genuinely save me money."
    )

    def test_exact_and_near_duplicates_collapse(self):
        a = single_turn("dlg-a", self.LU, self.AS)
        b = single_turn("dlg-b", self.LU, self.AS)  # exact duplicate
        c = single_turn("dlg-c", self.LU, self.AS + " and we should proceed soon")  # near (~0.94)
        d = single_turn(
            "dlg-d",
            "Hello, I would like to order a large pepperoni pizza for delivery tonight please.",
            "Sure thing, it is on its way to you right now.",
        )  # unrelated
        kept, stats = minhash_dedup([a, b, c, d], threshold=0.85, seed=42)
        kept_ids = {r.id for r in kept}
        assert "dlg-d" in kept_ids  # unrelated always survives
        assert len(kept) == 2  # {a,b,c} collapse to one + d
        assert stats.n_dropped == 2
        assert stats.n_clusters == 1

    def test_keep_priority_prefers_real_over_synthetic(self):
        # Identical text -> exact duplicates; keep-priority must pick the real one.
        synth = single_turn("dlg-synth", self.LU, self.AS, source="synthetic:v1")
        real = single_turn("dlg-real", self.LU, self.AS, source="hf:ds")
        kept, _ = minhash_dedup([synth, real], threshold=0.85, seed=42)
        assert len(kept) == 1
        assert kept[0].id == "dlg-real"

    def test_keep_priority_prefers_longer_when_both_real(self):
        short = single_turn("dlg-short", self.LU, self.AS, source="hf:ds")
        long_ = single_turn("dlg-long", self.LU, self.AS + " truly indeed", source="hf:ds")
        kept, _ = minhash_dedup([short, long_], threshold=0.85, seed=42)
        assert len(kept) == 1
        assert kept[0].id == "dlg-long"

    def test_idempotent(self):
        recs = [
            single_turn("dlg-a", "I am calling from ABC Energy about your electricity bill today."),
            single_turn("dlg-b", "I am calling from ABC Energy about your electricity bill today."),
            single_turn("dlg-c", "Totally unrelated message about booking a dentist appointment."),
        ]
        once, s1 = minhash_dedup(recs, threshold=0.85, seed=42)
        twice, s2 = minhash_dedup(once, threshold=0.85, seed=42)
        assert [r.id for r in once] == [r.id for r in twice]
        assert s1.n_dropped == 1
        assert s2.n_dropped == 0  # nothing left to drop

    def test_survivors_in_input_order(self):
        recs = [
            single_turn("dlg-1", "Energy switch pitch number one about your monthly power bill."),
            single_turn("dlg-2", "A completely different subject regarding car insurance quotes."),
            single_turn("dlg-1b", "Energy switch pitch number one about your monthly power bill."),
        ]
        kept, _ = minhash_dedup(recs, threshold=0.85, seed=42)
        ids = [r.id for r in kept]
        assert ids == sorted(ids, key=lambda x: [r.id for r in recs].index(x))


# ---------------------------------------------------------------------------
# T3.0 -- energy matcher + downsample
# ---------------------------------------------------------------------------


class TestEnergyMatcher:
    def test_word_boundary_not_substring(self):
        m = compile_energy_matcher(["electric", "provider"])
        assert m.search("we sell electric vehicles")  # exact word
        assert not m.search("I called an electrician yesterday")  # no substring hit
        # 'provider' is intentionally NOT a configured generic keyword in prod,
        # but the matcher itself is still word-boundary correct:
        assert m.search("your energy provider")
        assert not m.search("providerless setup")

    def test_multiword_and_hyphen(self):
        m = compile_energy_matcher(["power bill", "off-peak", "kwh"])
        assert m.search("my POWER BILL was high")
        assert m.search("cheaper Off-Peak rates")
        assert m.search("we used 500 kWh last month")
        assert not m.search("powerball lottery ticket")

    def test_empty_keywords_matches_nothing(self):
        m = compile_energy_matcher([])
        assert not m.search("energy electricity kwh")


ENERGY_KW = ["energy", "electricity", "kwh", "power bill", "tariff", "solar"]
HV_SCENARIOS = ["objection_handling", "info_gathering"]


def _general_pool(n: int, prefix: str = "gen") -> list[DialogueRecord]:
    return [
        single_turn(f"dlg-{prefix}-{i:04d}", f"Generic chit-chat number {i} about smartphones.")
        for i in range(n)
    ]


class TestDownsample:
    def test_high_value_always_kept_and_fill_to_target(self):
        # 3 label-only, 2 keyword-only, 1 both, plus a big general pool.
        recs = [
            single_turn("dlg-lab-1", "I am not interested.", scenario="objection_handling"),
            single_turn("dlg-lab-2", "How much do you pay?", scenario="info_gathering"),
            single_turn("dlg-lab-3", "Tell me more.", scenario="objection_handling"),
            single_turn("dlg-kw-1", "My power bill is high.", scenario="general"),
            single_turn("dlg-kw-2", "What about solar panels?", scenario="general"),
            single_turn(
                "dlg-both-1", "I cannot afford this tariff.", scenario="objection_handling"
            ),
        ] + _general_pool(50)
        kept, rep = downsample_m1(
            recs, m2_count=10, ratio=2.0, energy_keywords=ENERGY_KW,
            high_value_scenarios=HV_SCENARIOS, seed=42,
        )
        kept_ids = {r.id for r in kept}
        # every high-value record present
        for hv in ["dlg-lab-1", "dlg-lab-2", "dlg-lab-3", "dlg-kw-1", "dlg-kw-2", "dlg-both-1"]:
            assert hv in kept_ids
        assert rep.target_m1 == 20
        assert rep.final_m1 == 20
        assert rep.label_only == 3
        assert rep.keyword_only == 2
        assert rep.both == 1
        assert rep.high_value_union == 6
        assert rep.energy_keyword_hits == 3  # kw-1, kw-2, both-1
        assert rep.filled_general == 14
        assert not rep.high_value_exceeds_target

    def test_ratio_variants(self):
        recs = [
            single_turn("dlg-lab-1", "Not interested.", scenario="objection_handling"),
        ] + _general_pool(100)
        # ratio 3.0 -> target 30
        _, r3 = downsample_m1(recs, 10, 3.0, ENERGY_KW, HV_SCENARIOS, seed=42)
        assert r3.target_m1 == 30 and r3.final_m1 == 30
        # ratio 0 -> M2-only, drop all M1
        kept0, r0 = downsample_m1(recs, 10, 0.0, ENERGY_KW, HV_SCENARIOS, seed=42)
        assert kept0 == [] and r0.final_m1 == 0 and r0.target_m1 == 0
        assert not r0.high_value_exceeds_target

    def test_high_value_exceeds_target_keeps_all_high_value(self):
        # 5 high-value, tiny target -> keep all 5, add no general, flag set.
        recs = [
            single_turn(f"dlg-lab-{i}", "Not interested.", scenario="objection_handling")
            for i in range(5)
        ] + _general_pool(20)
        kept, rep = downsample_m1(recs, m2_count=1, ratio=2.0, energy_keywords=ENERGY_KW,
                                  high_value_scenarios=HV_SCENARIOS, seed=42)
        assert rep.target_m1 == 2
        assert rep.high_value_union == 5
        assert rep.final_m1 == 5  # all high-value kept, target breached upward
        assert rep.filled_general == 0
        assert rep.high_value_exceeds_target
        assert all(r.scenario == "objection_handling" for r in kept)

    def test_idempotent_same_seed(self):
        recs = [
            single_turn("dlg-lab-1", "Not interested.", scenario="objection_handling"),
        ] + _general_pool(100)
        a, _ = downsample_m1(recs, 10, 2.0, ENERGY_KW, HV_SCENARIOS, seed=42)
        b, _ = downsample_m1(recs, 10, 2.0, ENERGY_KW, HV_SCENARIOS, seed=42)
        assert [r.id for r in a] == [r.id for r in b]

    def test_different_seed_changes_general_sample(self):
        recs = [
            single_turn("dlg-lab-1", "Not interested.", scenario="objection_handling"),
        ] + _general_pool(100)
        a, _ = downsample_m1(recs, 10, 2.0, ENERGY_KW, HV_SCENARIOS, seed=1)
        b, _ = downsample_m1(recs, 10, 2.0, ENERGY_KW, HV_SCENARIOS, seed=2)
        assert {r.id for r in a} != {r.id for r in b}

    def test_union_is_or_not_and(self):
        # keyword hit in a non-high-value scenario must still be kept (orthogonal axes).
        recs = [single_turn("dlg-kw", "My electricity tariff is steep.", scenario="general")]
        recs += _general_pool(5)
        kept, rep = downsample_m1(recs, 10, 2.0, ENERGY_KW, HV_SCENARIOS, seed=42)
        assert "dlg-kw" in {r.id for r in kept}
        assert rep.keyword_only == 1 and rep.label_only == 0


# ---------------------------------------------------------------------------
# T3.2 -- stratified_split
# ---------------------------------------------------------------------------


def make_split_corpus() -> list[DialogueRecord]:
    """200 records across two scenarios x short bucket (large strata)."""
    recs = []
    for i in range(120):
        recs.append(
            single_turn(f"dlg-obj-{i:04d}", f"Objection sample {i}.", scenario="objection_handling")
        )
    for i in range(80):
        recs.append(
            single_turn(f"dlg-info-{i:04d}", f"Info sample {i}.", scenario="info_gathering")
        )
    return recs


class TestStratifiedSplit:
    def test_empty(self):
        splits, meta = stratified_split([], (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        assert splits == {"train": [], "val": [], "test": []}

    def test_ratios_and_total_preserved(self):
        recs = make_split_corpus()
        splits, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(recs)
        assert abs(len(splits["train"]) / total - 0.9) < 0.02
        assert abs(len(splits["val"]) / total - 0.05) < 0.02
        assert abs(len(splits["test"]) / total - 0.05) < 0.02

    def test_no_id_appears_in_two_splits(self):
        recs = make_split_corpus()
        splits, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        ids = [r.id for s in splits.values() for r in s]
        assert len(ids) == len(set(ids))

    def test_stratum_proportions_match_across_splits(self):
        recs = make_split_corpus()
        splits, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        # objection is 60% of the corpus; each split should be close.
        for s in splits.values():
            frac = sum(1 for r in s if r.scenario == "objection_handling") / len(s)
            assert abs(frac - 0.6) < 0.12

    def test_idempotent_same_seed(self):
        recs = make_split_corpus()
        a, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        b, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        for k in a:
            assert [r.id for r in a[k]] == [r.id for r in b[k]]

    def test_order_independent_same_seed(self):
        recs = make_split_corpus()
        a, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        b, _ = stratified_split(
            list(reversed(recs)), (0.9, 0.05, 0.05), default_stratum_key, seed=42
        )
        for k in a:
            assert {r.id for r in a[k]} == {r.id for r in b[k]}

    def test_small_strata_are_merged(self):
        # A scenario with tiny short(5) + mid(8) buckets must merge, not vanish.
        recs = make_split_corpus()
        for i in range(5):
            recs.append(
                make_record(
                    f"dlg-cls-s-{i}", ("user", "x"), ("assistant", "y"),
                    scenario="closing", n_turns=1,
                )
            )
        for i in range(8):
            # n_turns=6 -> mid bucket
            msgs = [("user", "a"), ("assistant", "b")] * 3
            recs.append(make_record(f"dlg-cls-m-{i}", *msgs, scenario="closing", n_turns=3))
        splits, meta = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        # all 13 closing records retained somewhere
        closing = sum(
            1 for s in splits.values() for r in s if r.scenario == "closing"
        )
        assert closing == 13
        # the two small closing buckets got merged into one effective stratum
        assert any(len(v) > 1 for v in meta.merged.values())

    def test_tiny_scenario_folds_into_residual_or_largest(self):
        # A 3-record scenario (< min_stratum) must still be split, not dropped.
        recs = make_split_corpus()
        for i in range(3):
            recs.append(single_turn(f"dlg-rare-{i}", "rare", scenario="cold_open"))
        splits, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        total = sum(len(s) for s in splits.values())
        assert total == len(recs)


# ---------------------------------------------------------------------------
# T3.3 -- leakage assertion
# ---------------------------------------------------------------------------


class TestLeakage:
    def test_clean_splits_have_no_leak(self):
        splits = {
            "train": [single_turn(f"dlg-t-{i}", f"Train unique message {i} about energy plans.")
                      for i in range(10)],
            "val": [single_turn("dlg-v", "Validation only message about pizza delivery times.")],
            "test": [single_turn("dlg-x", "Test only message about booking a flight to Tokyo.")],
        }
        rep = assert_no_leakage(splits, threshold=0.85, seed=42)
        assert rep.cross_split_dups == 0
        assert rep.as_dict()["cross_split_dups"] == 0

    def test_injected_cross_split_dup_is_detected(self):
        shared = (
            "I am calling from ABC Energy about switching your electricity supplier to "
            "lower your monthly power bill before the cold winter season truly arrives."
        )
        splits = {
            "train": [single_turn("dlg-train", shared)],
            "val": [single_turn("dlg-val", shared)],  # leaked near-duplicate
            "test": [single_turn("dlg-test", "Unrelated content about gardening tips for spring.")],
        }
        rep = assert_no_leakage(splits, threshold=0.85, seed=42)
        assert rep.cross_split_dups >= 1
        assert rep.examples  # an example pair is reported

    def test_same_split_duplicates_are_not_counted(self):
        shared = "I am calling from ABC Energy about your electricity bill and plan today friend."
        splits = {
            "train": [single_turn("dlg-a", shared), single_turn("dlg-b", shared)],
            "val": [],
            "test": [single_turn("dlg-c", "A totally different topic, namely scuba diving.")],
        }
        rep = assert_no_leakage(splits, threshold=0.85, seed=42)
        assert rep.cross_split_dups == 0


# ---------------------------------------------------------------------------
# Distribution reporting
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_scenario_distribution_sums_to_one(self):
        recs = make_split_corpus()
        dist = scenario_distribution(recs)
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        assert abs(dist["objection_handling"] - 0.6) < 0.01

    def test_distribution_warnings_flags_skew(self):
        # val is 100% info_gathering while overall is mixed -> deviation > 3pp.
        splits = {
            "train": make_split_corpus(),
            "val": [single_turn(f"dlg-skew-{i}", "x", scenario="info_gathering") for i in range(5)],
            "test": [],
        }
        warns = distribution_warnings(splits, default_stratum_key, warn_pp=3.0)
        assert any(w["split"] == "val" for w in warns)

    def test_no_warnings_when_balanced(self):
        recs = make_split_corpus()
        splits, _ = stratified_split(recs, (0.9, 0.05, 0.05), default_stratum_key, seed=42)
        warns = distribution_warnings(splits, default_stratum_key, warn_pp=3.0)
        # well-stratified large strata -> no deviation above 3pp
        assert warns == []
