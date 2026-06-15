"""Unit tests for M7 calibration-set construction and size accounting (T7.1/T7.3).

Pure logic, no GPU: sampling count/stratification/determinism, ChatML rendering
format, and the FP16-vs-INT4 size arithmetic.
"""

from __future__ import annotations

import json

from sales_agent.common.schema import DialogueRecord
from sales_agent.quant.calibration import (
    CalibrationReport,
    load_calibration_texts,
    model_dir_size_bytes,
    render_calibration_text,
    size_report,
    stratified_calibration_sample,
)


def make_record(idx: int, scenario: str, n_turns: int) -> DialogueRecord:
    """A valid alternating user/assistant dialogue with ``n_turns`` assistant turns."""
    messages = []
    for t in range(n_turns):
        messages.append({"role": "user", "content": f"q{t} about {scenario}"})
        messages.append({"role": "assistant", "content": f"a{t} reply for {scenario}"})
    return DialogueRecord(
        id=f"dlg-{scenario}-{n_turns}-{idx:04d}",
        source="synthetic:v1",
        scenario=scenario,
        lang="en",
        n_turns=n_turns,
        messages=messages,
    )


def build_corpus() -> list[DialogueRecord]:
    """A skewed corpus mirroring train: general-heavy, all 5 scenarios, 3 buckets."""
    spec = {
        "general": 120,
        "objection_handling": 40,
        "info_gathering": 26,
        "cold_open": 10,
        "closing": 10,
    }
    recs: list[DialogueRecord] = []
    i = 0
    for scenario, count in spec.items():
        for j in range(count):
            n_turns = (3, 6, 10)[j % 3]  # short / mid / long buckets
            recs.append(make_record(i, scenario, n_turns))
            i += 1
    return recs


# ---------------------------------------------------------------------------
# Sampling: count, proportionality, coverage, determinism
# ---------------------------------------------------------------------------


def test_sample_hits_exact_count():
    recs = build_corpus()
    selected, report = stratified_calibration_sample(recs, n_samples=64, seed=42)
    assert len(selected) == 64
    assert report.n_selected == 64
    assert report.n_requested == 64
    assert report.n_input == len(recs)
    assert sum(report.per_stratum.values()) == 64
    assert sum(report.per_scenario.values()) == 64


def test_sample_is_strictly_proportional():
    # general is ~57% of the corpus; with strict proportional allocation it should
    # dominate the sample at roughly the same share (user decision: no coverage floor).
    recs = build_corpus()
    n = 100
    selected, report = stratified_calibration_sample(recs, n_samples=n, seed=42)
    general_share_corpus = sum(r.scenario == "general" for r in recs) / len(recs)
    general_share_sample = report.per_scenario["general"] / n
    assert abs(general_share_sample - general_share_corpus) < 0.05


def test_all_scenarios_covered_at_256_equivalent():
    # Even strict-proportional, every scenario survives when the sample is large
    # enough relative to the smallest stratum.
    recs = build_corpus()
    _, report = stratified_calibration_sample(recs, n_samples=120, seed=42)
    assert set(report.per_scenario) == {
        "general",
        "objection_handling",
        "info_gathering",
        "cold_open",
        "closing",
    }
    assert all(v > 0 for v in report.per_scenario.values())


def test_sampling_deterministic_for_seed():
    recs = build_corpus()
    a, _ = stratified_calibration_sample(recs, 50, seed=42)
    b, _ = stratified_calibration_sample(recs, 50, seed=42)
    assert [r.id for r in a] == [r.id for r in b]


def test_sampling_changes_with_seed():
    recs = build_corpus()
    a, _ = stratified_calibration_sample(recs, 50, seed=42)
    b, _ = stratified_calibration_sample(recs, 50, seed=7)
    # Same per-stratum counts (proportional alloc is seed-independent), different picks.
    assert [r.id for r in a] != [r.id for r in b]


def test_sampling_order_independent():
    # Shuffling the input must not change the selected set (sorted-by-id base).
    recs = build_corpus()
    a, _ = stratified_calibration_sample(recs, 50, seed=42)
    b, _ = stratified_calibration_sample(list(reversed(recs)), 50, seed=42)
    assert {r.id for r in a} == {r.id for r in b}


def test_request_exceeds_corpus_returns_all():
    recs = build_corpus()
    selected, report = stratified_calibration_sample(recs, n_samples=10_000, seed=42)
    assert len(selected) == len(recs)
    assert report.n_selected == len(recs)


def test_empty_corpus():
    selected, report = stratified_calibration_sample([], n_samples=32, seed=42)
    assert selected == []
    assert report.n_selected == 0


def test_smoke_count_32():
    recs = build_corpus()
    selected, _ = stratified_calibration_sample(recs, n_samples=32, seed=42)
    assert len(selected) == 32


# ---------------------------------------------------------------------------
# Rendering format
# ---------------------------------------------------------------------------


def test_render_is_chatml_with_assistant_and_no_think():
    rec = make_record(0, "closing", 2)
    text = render_calibration_text(rec)
    assert text.startswith("<|im_start|>user\n")
    assert "<|im_start|>assistant\n" in text
    assert text.count("<|im_end|>") == 4  # 2 user + 2 assistant turns
    # No <think> block is injected (render_chatml mirrors the Instruct template).
    assert "<think>" not in text
    # No trailing generation prompt (full conversation is the calibration sample).
    assert not text.endswith("<|im_start|>assistant\n")


def test_render_includes_system_when_present():
    rec = DialogueRecord(
        id="dlg-sys-0001",
        source="synthetic:v1",
        scenario="general",
        lang="en",
        n_turns=1,
        messages=[
            {"role": "system", "content": "You are an ABC Energy agent."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ],
    )
    text = render_calibration_text(rec)
    assert text.startswith("<|im_start|>system\nYou are an ABC Energy agent.")


def test_load_calibration_texts_from_jsonl(tmp_path):
    recs = build_corpus()
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r.model_dump()) + "\n")
    texts, report = load_calibration_texts(path, n_samples=40, seed=42)
    assert len(texts) == 40
    assert all(t.startswith("<|im_start|>") for t in texts)
    assert isinstance(report, CalibrationReport)
    assert report.n_selected == 40


# ---------------------------------------------------------------------------
# Size accounting (FP16 vs INT4)
# ---------------------------------------------------------------------------


def test_size_report_arithmetic():
    rep = size_report(fp16_bytes=8_000_000_000, int4_bytes=2_000_000_000)
    assert rep["fp16_gb"] == 8.0
    assert rep["int4_gb"] == 2.0
    assert rep["compression_ratio"] == 4.0
    assert rep["size_reduction_pct"] == 75.0


def test_size_report_handles_zero_int4():
    rep = size_report(fp16_bytes=8_000_000_000, int4_bytes=0)
    assert rep["compression_ratio"] is None
    assert rep["size_reduction_pct"] == 100.0


def test_model_dir_size_bytes(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"y" * 500)
    (tmp_path / "config.json").write_text("{}")  # non-weight files ignored
    assert model_dir_size_bytes(tmp_path) == 1500
