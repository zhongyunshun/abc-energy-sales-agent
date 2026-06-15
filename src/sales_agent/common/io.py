"""JSONL read/write helpers (UTF-8, one object per line)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one parsed object per non-blank line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Write records as JSONL, creating parent dirs. Returns record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
