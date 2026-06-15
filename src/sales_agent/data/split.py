"""M3 split & leakage prevention: global near-dedup, M1 downsampling,
stratified train/val/test split, and a cross-split leakage assertion
(the M3 split and leakage contract).

Pure-CPU logic, no GPU and no API. Everything here is unit-testable; the thin
CLI in ``scripts/data/split.py`` wires these functions in the mandated runtime
order: exact dedup -> global MinHash near-dedup -> M1 downsample -> stratified
split -> leakage assertion.

Reuses M1 hashing helpers from :mod:`sales_agent.data.dedup`
(``normalize_text``, ``content_hash``, ``dedup_exact``) rather than
reimplementing them.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from sales_agent.common.schema import DialogueRecord
from sales_agent.data.dedup import normalize_text

# ---------------------------------------------------------------------------
# Turn bucketing & stratify key (the M3 contract)
# ---------------------------------------------------------------------------

TURN_BUCKETS = ("short", "mid", "long")  # ordered short < mid < long


def turn_bucket(n_turns: int) -> str:
    """Map assistant-turn count to a bucket: short<=4 | mid 5-8 | long>8."""
    if n_turns <= 4:
        return "short"
    if n_turns <= 8:
        return "mid"
    return "long"


def default_stratum_key(rec: DialogueRecord) -> tuple[str, str]:
    """Stratify key ``(scenario, turn_bucket)`` used by the split."""
    return (rec.scenario, turn_bucket(rec.n_turns))


def is_real(rec: DialogueRecord) -> bool:
    """True for real/public records, False for synthetic (``source`` prefix)."""
    return not rec.source.lower().startswith("synthetic")


def _dialogue_length(rec: DialogueRecord) -> int:
    """Total character length of all message contents (cluster keep-priority)."""
    return sum(len(m.content) for m in rec.messages)


# ---------------------------------------------------------------------------
# T3.1 -- global near-duplicate dedup (MinHash + LSH)
# ---------------------------------------------------------------------------


@dataclass
class DedupStats:
    """Outcome of one :func:`minhash_dedup` pass."""

    n_input: int
    n_kept: int
    n_dropped: int
    n_clusters: int  # number of clusters with more than one member
    threshold: float


def _shingles(text: str, k: int) -> set[str]:
    """Word-level k-gram shingles; short texts collapse to a single shingle."""
    tokens = text.split()
    if len(tokens) < k:
        return {text} if text else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _build_minhash(text: str, num_perm: int, seed: int, shingle_size: int) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=seed)
    shingles = _shingles(text, shingle_size)
    if shingles:
        m.update_batch([s.encode("utf-8") for s in shingles])
    return m


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # keep smaller index as root


def minhash_dedup(
    records: list[DialogueRecord],
    threshold: float,
    seed: int,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> tuple[list[DialogueRecord], DedupStats]:
    """Drop near-duplicate dialogues globally (before any split).

    Pipeline: whole-dialogue normalized text -> word ``shingle_size``-gram
    shingles -> MinHash(``num_perm``, seeded) -> LSH banding at ``threshold``
    -> verify candidate pairs by estimated Jaccard -> union-find clusters ->
    keep one record per cluster.

    Cluster keep-priority (the keep-priority rule): prefer real over synthetic, then the
    longer dialogue, then the earliest input index (deterministic tie-break).
    Survivors are returned in original input order, so a second pass drops
    nothing (idempotent).
    """
    n = len(records)
    if n == 0:
        return [], DedupStats(0, 0, 0, 0, threshold)

    mhs = [
        _build_minhash(normalize_text(r.messages), num_perm, seed, shingle_size) for r in records
    ]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(mhs):
        lsh.insert(str(i), m)

    uf = _UnionFind(n)
    for i, m in enumerate(mhs):
        for cand in lsh.query(m):
            j = int(cand)
            if j <= i:
                continue
            if mhs[i].jaccard(mhs[j]) >= threshold:
                uf.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    kept_indices: list[int] = []
    n_multi = 0
    for members in clusters.values():
        if len(members) > 1:
            n_multi += 1
        best = max(
            members,
            key=lambda idx: (is_real(records[idx]), _dialogue_length(records[idx]), -idx),
        )
        kept_indices.append(best)

    kept_indices.sort()  # restore input order
    kept = [records[i] for i in kept_indices]
    return kept, DedupStats(n, len(kept), n - len(kept), n_multi, threshold)


# ---------------------------------------------------------------------------
# T3.0 -- M1 downsampling (after global dedup, before split)
# ---------------------------------------------------------------------------


@dataclass
class DownsampleReport:
    """Reproducible breakdown of the M1 downsampling decision."""

    m2_count: int
    ratio: float
    target_m1: int
    n_input_m1: int
    label_only: int  # high-value via scenario label axis only
    keyword_only: int  # high-value via energy-keyword axis only
    both: int  # high-value via both axes
    energy_keyword_hits: int  # records matching the tightened energy list
    high_value_union: int
    filled_general: int  # records randomly sampled from the remaining general pool
    final_m1: int
    m1_to_m2_ratio: float
    high_value_exceeds_target: bool

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def compile_energy_matcher(keywords: list[str]) -> re.Pattern[str]:
    """Compile a case-insensitive word-boundary matcher over ``keywords``.

    Multi-word and hyphenated phrases (e.g. ``"power bill"``, ``"off-peak"``)
    are supported: the ``\\b`` anchors wrap the whole alternation, so each
    phrase is matched at token boundaries rather than as a loose substring.
    This is the tightened replacement for substring probing, which over-fires
    on generic words (``provider`` in "service provider", ``electric`` in
    "electrician").
    """
    if not keywords:
        # Match nothing.
        return re.compile(r"(?!x)x")
    alternation = "|".join(re.escape(k.lower()) for k in keywords)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def _full_text(rec: DialogueRecord) -> str:
    return " ".join(m.content for m in rec.messages)


def downsample_m1(
    m1_records: list[DialogueRecord],
    m2_count: int,
    ratio: float,
    energy_keywords: list[str],
    high_value_scenarios: list[str],
    seed: int,
) -> tuple[list[DialogueRecord], DownsampleReport]:
    """Downsample M1 so it does not drown M2's multi-turn energy dialogues.

    High-value set (always kept, never dropped to hit a ratio) is the UNION of
    two orthogonal axes:
      (a) records whose ``scenario`` is in ``high_value_scenarios``;
      (b) records whose full dialogue text matches an energy keyword
          (word-boundary).
    The remainder ("general" pool) is randomly sampled with a fixed ``seed`` to
    top the kept set up to ``target = round(ratio × m2_count)``.

    ``ratio == 0`` means "M2-only": keep nothing from M1. If the high-value
    union already meets/exceeds the target, all high-value records are kept and
    no general records are added (``high_value_exceeds_target`` is set; the CLI
    surfaces this for confirmation rather than discarding high-value data).

    Returns the kept records (in original input order) and a reproducible
    :class:`DownsampleReport`.
    """
    n_input = len(m1_records)
    target = round(ratio * m2_count)
    matcher = compile_energy_matcher(energy_keywords)
    hv_scenarios = set(high_value_scenarios)

    label_idx: set[int] = set()
    keyword_idx: set[int] = set()
    for i, rec in enumerate(m1_records):
        if rec.scenario in hv_scenarios:
            label_idx.add(i)
        if matcher.search(_full_text(rec)):
            keyword_idx.add(i)

    union_idx = label_idx | keyword_idx
    general_idx = [i for i in range(n_input) if i not in union_idx]

    high_value_exceeds = len(union_idx) >= target and target > 0
    if ratio == 0:
        kept_idx: set[int] = set()
        filled = 0
    elif high_value_exceeds:
        kept_idx = set(union_idx)
        filled = 0
    else:
        need = target - len(union_idx)
        rng = random.Random(seed)
        # Sort for input-order independence before the seeded sample.
        pool = sorted(general_idx, key=lambda i: m1_records[i].id)
        sampled = rng.sample(pool, k=min(need, len(pool)))
        kept_idx = set(union_idx) | set(sampled)
        filled = len(sampled)

    kept = [m1_records[i] for i in sorted(kept_idx)]
    final_m1 = len(kept)
    report = DownsampleReport(
        m2_count=m2_count,
        ratio=ratio,
        target_m1=target,
        n_input_m1=n_input,
        label_only=len(label_idx - keyword_idx),
        keyword_only=len(keyword_idx - label_idx),
        both=len(label_idx & keyword_idx),
        energy_keyword_hits=len(keyword_idx),
        high_value_union=len(union_idx),
        filled_general=filled,
        final_m1=final_m1,
        m1_to_m2_ratio=round(final_m1 / m2_count, 4) if m2_count else 0.0,
        high_value_exceeds_target=high_value_exceeds and ratio != 0,
    )
    return kept, report


# ---------------------------------------------------------------------------
# T3.2 -- stratified split with small-stratum merging
# ---------------------------------------------------------------------------


@dataclass
class SplitMeta:
    """How records were grouped into the strata that were actually split."""

    effective_strata: dict[str, int] = field(default_factory=dict)  # label -> count
    merged: dict[str, list[str]] = field(default_factory=dict)  # label -> raw keys merged


def _stratum_int_seed(seed: int, label: str) -> int:
    """Deterministic per-stratum seed (sha256-based; not Python ``hash``).

    ``hash(str)`` is salted per process (PYTHONHASHSEED) and would break
    reproducibility across runs, so we derive the seed from a stable digest.
    """
    h = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    return int(h[:16], 16)


def _merge_small_strata(
    groups: dict[tuple[str, str], list[DialogueRecord]],
    min_stratum: int,
) -> tuple[dict[str, list[DialogueRecord]], SplitMeta]:
    """Merge strata smaller than ``min_stratum`` to avoid empty/degenerate layers.

    Two phases (the small-layer merge rule: "small layers merge into a neighboring bucket"):
      1. Within each scenario, merge adjacent turn buckets (short<mid<long)
         left-to-right until each chunk reaches ``min_stratum``; a small tail
         folds back into the previous chunk.
      2. Any scenario still below ``min_stratum`` is pooled into a shared
         ``misc|all`` residual; if that residual is itself too small it folds
         into the currently largest stratum.
    """
    by_scenario: dict[str, dict[str, list[DialogueRecord]]] = defaultdict(dict)
    for (scenario, bucket), recs in groups.items():
        by_scenario[scenario][bucket] = recs

    effective: dict[str, list[DialogueRecord]] = {}
    merged_map: dict[str, list[str]] = {}
    residual: list[DialogueRecord] = []
    residual_keys: list[str] = []

    for scenario in sorted(by_scenario):
        buckets = by_scenario[scenario]
        ordered = [b for b in TURN_BUCKETS if b in buckets]
        chunks: list[tuple[list[str], list[DialogueRecord]]] = []
        cur_labels: list[str] = []
        cur_recs: list[DialogueRecord] = []
        for b in ordered:
            cur_labels.append(b)
            cur_recs = cur_recs + buckets[b]
            if len(cur_recs) >= min_stratum:
                chunks.append((cur_labels, cur_recs))
                cur_labels, cur_recs = [], []
        if cur_recs:  # small tail
            if chunks:
                prev_labels, prev_recs = chunks[-1]
                chunks[-1] = (prev_labels + cur_labels, prev_recs + cur_recs)
            else:
                chunks.append((cur_labels, cur_recs))

        for labels, recs in chunks:
            raw_keys = [f"{scenario}|{b}" for b in labels]
            if len(recs) < min_stratum:
                # Whole scenario too small -> defer to cross-scenario residual.
                residual.extend(recs)
                residual_keys.extend(raw_keys)
            else:
                label = f"{scenario}|{'+'.join(labels)}"
                effective[label] = recs
                merged_map[label] = raw_keys

    if residual:
        if len(residual) >= min_stratum or not effective:
            effective["misc|all"] = residual
            merged_map["misc|all"] = residual_keys
        else:
            # Fold into the largest existing stratum.
            biggest = max(effective, key=lambda k: len(effective[k]))
            effective[biggest] = effective[biggest] + residual
            merged_map[biggest] = merged_map[biggest] + residual_keys

    meta = SplitMeta(
        effective_strata={k: len(v) for k, v in effective.items()},
        merged={k: v for k, v in merged_map.items() if len(v) > 1},
    )
    return effective, meta


def stratified_split(
    records: list[DialogueRecord],
    ratios: tuple[float, float, float],
    key_fn: Callable[[DialogueRecord], tuple[str, str]],
    seed: int,
    min_stratum: int = 20,
) -> tuple[dict[str, list[DialogueRecord]], SplitMeta]:
    """Split whole dialogues into train/val/test, stratified by ``key_fn``.

    Records are grouped by ``key_fn`` (default ``(scenario, turn_bucket)``);
    small strata are merged (:func:`_merge_small_strata`); each effective
    stratum is shuffled with a deterministic per-stratum seed and sliced by
    cumulative ``ratios`` so per-stratum counts always sum exactly. Splitting
    is by complete dialogue (never by turn). Same input + same seed is
    idempotent.
    """
    splits: dict[str, list[DialogueRecord]] = {"train": [], "val": [], "test": []}
    if not records:
        return splits, SplitMeta()

    groups: dict[tuple[str, str], list[DialogueRecord]] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)

    effective, meta = _merge_small_strata(groups, min_stratum)

    r_train, r_val = ratios[0], ratios[1]
    for label in sorted(effective):
        recs = sorted(effective[label], key=lambda r: r.id)  # order-independent base
        rng = random.Random(_stratum_int_seed(seed, label))
        rng.shuffle(recs)
        n = len(recs)
        c1 = round(n * r_train)
        c2 = round(n * (r_train + r_val))
        splits["train"].extend(recs[:c1])
        splits["val"].extend(recs[c1:c2])
        splits["test"].extend(recs[c2:])

    return splits, meta


# ---------------------------------------------------------------------------
# T3.3 -- cross-split leakage assertion (acceptance #3)
# ---------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """Result of the cross-split near-duplicate check."""

    method: str
    threshold: float
    cross_split_dups: int
    examples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "cross_split_dups": self.cross_split_dups,
        }


def assert_no_leakage(
    splits: dict[str, list[DialogueRecord]],
    threshold: float,
    seed: int,
    num_perm: int = 128,
    shingle_size: int = 5,
    max_examples: int = 10,
) -> LeakageReport:
    """Count near-duplicate dialogue pairs that straddle two different splits.

    Builds one MinHash per record (same parameters as :func:`minhash_dedup`),
    inserts them into a single LSH keyed by ``"split:index"``, and counts
    candidate pairs from *different* splits whose estimated Jaccard meets
    ``threshold``. A non-zero count is a leak; the CLI turns that into exit
    code 2. After a correct global near-dedup before splitting, this is 0.
    """
    items: list[tuple[str, int, DialogueRecord]] = []
    for split_name, recs in splits.items():
        for i, rec in enumerate(recs):
            items.append((split_name, i, rec))

    if not items:
        return LeakageReport("minhash", threshold, 0)

    mhs = [_build_minhash(normalize_text(rec.messages), num_perm, seed, shingle_size)
           for (_, _, rec) in items]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, m in enumerate(mhs):
        lsh.insert(str(idx), m)

    seen_pairs: set[tuple[int, int]] = set()
    examples: list[dict] = []
    for idx, m in enumerate(mhs):
        for cand in lsh.query(m):
            j = int(cand)
            if j <= idx:
                continue
            if items[idx][0] == items[j][0]:
                continue  # same split
            if mhs[idx].jaccard(mhs[j]) < threshold:
                continue
            pair = (idx, j)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if len(examples) < max_examples:
                a, b = items[idx], items[j]
                examples.append(
                    {
                        "a": {"split": a[0], "id": a[2].id},
                        "b": {"split": b[0], "id": b[2].id},
                        "jaccard": round(mhs[idx].jaccard(mhs[j]), 4),
                    }
                )

    return LeakageReport("minhash", threshold, len(seen_pairs), examples)


# ---------------------------------------------------------------------------
# Distribution consistency (the M3 distribution check: >3pp -> WARN)
# ---------------------------------------------------------------------------


def scenario_distribution(records: list[DialogueRecord]) -> dict[str, float]:
    """Fraction of records per scenario (split_report ``distribution`` field)."""
    if not records:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.scenario] += 1
    n = len(records)
    return {k: round(v / n, 4) for k, v in sorted(counts.items())}


def distribution_warnings(
    splits: dict[str, list[DialogueRecord]],
    key_fn: Callable[[DialogueRecord], tuple[str, str]],
    warn_pp: float = 3.0,
) -> list[dict]:
    """Flag strata whose per-split share deviates from the overall share by >``warn_pp``.

    Compares each split's share of every ``(scenario, turn_bucket)`` stratum to
    the pooled overall share; returns one record per deviation above the
    threshold (in percentage points).
    """
    all_recs = [r for recs in splits.values() for r in recs]
    if not all_recs:
        return []

    def shares(recs: list[DialogueRecord]) -> dict[tuple[str, str], float]:
        if not recs:
            return {}
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for r in recs:
            counts[key_fn(r)] += 1
        n = len(recs)
        return {k: v / n for k, v in counts.items()}

    overall = shares(all_recs)
    warnings: list[dict] = []
    threshold = warn_pp / 100.0
    for split_name, recs in splits.items():
        sp = shares(recs)
        for stratum, ov in overall.items():
            dev = abs(sp.get(stratum, 0.0) - ov)
            if dev > threshold:
                warnings.append(
                    {
                        "split": split_name,
                        "stratum": list(stratum),
                        "overall_share": round(ov, 4),
                        "split_share": round(sp.get(stratum, 0.0), 4),
                        "deviation_pp": round(dev * 100, 2),
                    }
                )
    return warnings
