"""Tests for common/io.py."""

from sales_agent.common.io import read_jsonl, write_jsonl


def test_roundtrip(tmp_path):
    records = [{"a": 1}, {"b": "text with unicode: café"}, {"nested": {"x": [1, 2]}}]
    path = tmp_path / "out" / "data.jsonl"
    n = write_jsonl(path, records)
    assert n == 3
    assert list(read_jsonl(path)) == records


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": 2}]


def test_write_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    assert write_jsonl(path, []) == 0
    assert list(read_jsonl(path)) == []


def test_unicode_not_escaped(tmp_path):
    path = tmp_path / "u.jsonl"
    write_jsonl(path, [{"s": "café"}])
    assert "café" in path.read_text(encoding="utf-8")
