"""JSONL reading, filtering, and text extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed records from a JSONL file, skipping malformed lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text(record: dict, min_text_len: int = 200) -> str | None:
    """Concatenate title + snippet + content. Return None if too short or duplicate."""
    if record.get("dedupe") == "duplicate":
        return None

    parts: list[str] = []
    for key in ("title", "snippet", "content"):
        val = record.get(key)
        if val:
            parts.append(val)

    text = "\n\n".join(parts)
    if len(text) < min_text_len:
        return None
    return text
