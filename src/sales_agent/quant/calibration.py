"""M7 AWQ calibration-set construction and FP16-vs-INT4 size accounting.

Pure logic only: this module imports NO ``torch`` / ``transformers`` /
``llmcompressor`` so its unit tests run on the CPU-only Windows host (the CPU-only testability contract). The GPU quantization (llm-compressor ``oneshot``) lives in
``scripts/quant/quantize_awq.py`` and runs inside the train container.

What this builds
----------------
AWQ needs a calibration set drawn from the *training domain* (the M7 quantization target). We sample dialogues from ``train.jsonl`` and render each to plain ChatML
text:

- **Sampling** is stratified by ``(scenario, turn_bucket)`` -- the exact stratify
  key M3 used -- reusing :func:`sales_agent.data.split.default_stratum_key`
  (imported, never modified). Allocation across strata is in **strict proportion**
  to the train distribution (user decision), made to sum to exactly ``n_samples``
  via the largest-remainder method, and is deterministic for a fixed ``seed``.
- **Rendering** reuses :func:`sales_agent.training.formatting.render_chatml` so the
  calibration text is byte-identical to the SFT/DPO chat format. ``render_chatml``
  injects no ``<think>`` block, so the empty-``<think>`` prefix question (M6 Option
  A) does not enter calibration; downstream stripping stays an M8/M9/M10 concern.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sales_agent.common.io import read_jsonl
from sales_agent.common.schema import DialogueRecord
from sales_agent.data.split import default_stratum_key  # reuse M3 stratify key (do NOT modify)
from sales_agent.training.formatting import render_chatml  # reuse SFT/DPO rendering

# ---------------------------------------------------------------------------
# Stratified proportional sampling (T7.1)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Reproducible breakdown of one calibration sample."""

    n_requested: int
    n_selected: int
    n_input: int
    seed: int
    per_stratum: dict[str, int] = field(default_factory=dict)  # "scenario|bucket" -> count
    per_scenario: dict[str, int] = field(default_factory=dict)  # scenario -> count

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _stratum_seed(seed: int, label: tuple[str, str]) -> int:
    """Deterministic per-stratum seed (sha256-based, not salted ``hash``).

    Mirrors the approach in :mod:`sales_agent.data.split`: ``hash(str)`` is salted
    per process (PYTHONHASHSEED) and would break cross-run reproducibility, so the
    seed is derived from a stable digest.
    """
    key = f"{seed}:{label[0]}|{label[1]}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def _largest_remainder(
    weights: dict[tuple[str, str], int], total: int
) -> dict[tuple[str, str], int]:
    """Allocate ``total`` units across keys proportional to integer ``weights``.

    Uses the largest-remainder method so allocations sum to exactly ``total``.
    Tie-breaks on the larger remainder, then on the key (deterministic). With
    ``total <= sum(weights)`` no allocation exceeds its weight, so the caller can
    sample without replacement.
    """
    sum_w = sum(weights.values())
    if sum_w == 0 or total <= 0:
        return dict.fromkeys(weights, 0)

    floors: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    for label, w in weights.items():
        exact = total * w / sum_w
        f = int(exact)
        floors[label] = f
        remainders.append((exact - f, label))

    leftover = total - sum(floors.values())
    remainders.sort(key=lambda t: (-t[0], t[1]))
    for i in range(leftover):
        floors[remainders[i][1]] += 1
    return floors


def stratified_calibration_sample(
    records: list[DialogueRecord],
    n_samples: int,
    seed: int,
    key_fn: Callable[[DialogueRecord], tuple[str, str]] = default_stratum_key,
) -> tuple[list[DialogueRecord], CalibrationReport]:
    """Sample ``n_samples`` dialogues, stratified by ``key_fn`` in strict proportion.

    Records are grouped by ``key_fn`` (default ``(scenario, turn_bucket)``); each
    stratum's quota is proportional to its size (largest-remainder, summing to
    exactly ``n_samples``); within a stratum the records are sorted by ``id``
    (input-order independent) and sampled with a deterministic per-stratum seed.

    When ``n_samples >= len(records)`` every record is returned (all of train is
    the calibration set). Selection order is grouped by sorted stratum key, so the
    result is fully reproducible for a fixed ``seed``.
    """
    n_input = len(records)
    if n_input == 0:
        return [], CalibrationReport(n_samples, 0, 0, seed)

    groups: dict[tuple[str, str], list[DialogueRecord]] = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)

    if n_samples >= n_input:
        alloc = {label: len(recs) for label, recs in groups.items()}
    else:
        alloc = _largest_remainder({label: len(recs) for label, recs in groups.items()}, n_samples)

    selected: list[DialogueRecord] = []
    per_stratum: dict[str, int] = {}
    per_scenario: dict[str, int] = {}
    for label in sorted(groups):
        k = alloc.get(label, 0)
        recs = sorted(groups[label], key=lambda r: r.id)
        if k >= len(recs):
            chosen = recs
        else:
            chosen = random.Random(_stratum_seed(seed, label)).sample(recs, k)
        selected.extend(chosen)
        per_stratum[f"{label[0]}|{label[1]}"] = len(chosen)
        per_scenario[label[0]] = per_scenario.get(label[0], 0) + len(chosen)

    report = CalibrationReport(
        n_requested=n_samples,
        n_selected=len(selected),
        n_input=n_input,
        seed=seed,
        per_stratum=dict(sorted(per_stratum.items())),
        per_scenario=dict(sorted(per_scenario.items())),
    )
    return selected, report


def render_calibration_text(record: DialogueRecord) -> str:
    """Render a dialogue to plain ChatML calibration text (reuses ``render_chatml``).

    No trailing generation prompt: the full conversation (system/user/assistant
    turns) is the calibration sample, matching the training-domain text the model
    saw during SFT/DPO.
    """
    return render_chatml(record.messages)


def load_calibration_texts(
    train_path: str | Path,
    n_samples: int,
    seed: int,
) -> tuple[list[str], CalibrationReport]:
    """Read ``train.jsonl``, sample, and render to calibration texts.

    Thin orchestration over :func:`stratified_calibration_sample` and
    :func:`render_calibration_text`; the only I/O here is reading the JSONL, so it
    stays unit-testable with a tiny fixture and needs no GPU.
    """
    records = [DialogueRecord.model_validate(d) for d in read_jsonl(train_path)]
    selected, report = stratified_calibration_sample(records, n_samples, seed)
    texts = [render_calibration_text(r) for r in selected]
    return texts, report


# ---------------------------------------------------------------------------
# FP16-vs-INT4 size accounting (T7.3 -> manifest / README trade-off)
# ---------------------------------------------------------------------------


def model_dir_size_bytes(path: str | Path) -> int:
    """Total bytes of the ``*.safetensors`` weight shards in a model directory."""
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))


def size_report(fp16_bytes: int, int4_bytes: int) -> dict:
    """FP16 vs INT4 size comparison for the manifest and README trade-off section.

    Returns raw bytes, GB (SI, /1e9 to match the merge manifest's ``merged_size_gb``),
    the compression ratio (FP16/INT4), and the percentage size reduction.
    """
    return {
        "fp16_bytes": fp16_bytes,
        "int4_bytes": int4_bytes,
        "fp16_gb": round(fp16_bytes / 1e9, 3),
        "int4_gb": round(int4_bytes / 1e9, 3),
        "compression_ratio": round(fp16_bytes / int4_bytes, 3) if int4_bytes else None,
        "size_reduction_pct": round((1 - int4_bytes / fp16_bytes) * 100, 2) if fp16_bytes else None,
    }
