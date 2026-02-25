"""Persist search results and maintain a search log."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path.home() / ".brave-search"


def _ensure_dirs(data_dir: Path) -> None:
    (data_dir / "results").mkdir(parents=True, exist_ok=True)


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def _timestamp_str(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H-%M-%S")


class ResultStore:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = data_dir

    def save_result(
        self,
        data: dict[str, Any],
        query: str,
        *,
        now: datetime | None = None,
    ) -> Path:
        if now is None:
            now = datetime.now(timezone.utc)
        _ensure_dirs(self.data_dir)
        filename = f"{_timestamp_str(now)}_{_slugify(query)}.json"
        path = self.data_dir / "results" / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return path

    def append_log(
        self,
        query: str,
        params: dict[str, Any],
        result_count: int,
        result_file: Path,
        *,
        now: datetime | None = None,
    ) -> None:
        if now is None:
            now = datetime.now(timezone.utc)
        _ensure_dirs(self.data_dir)
        entry = {
            "timestamp": now.isoformat(),
            "query": query,
            "count": params.get("count", 20),
            "result_count": result_count,
            "result_file": str(result_file),
        }
        log_path = self.data_dir / "search.log"
        with log_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Backward-compatible shims
def save_result(
    data: dict[str, Any],
    query: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> Path:
    return ResultStore(data_dir).save_result(data, query, now=now)


def append_log(
    query: str,
    params: dict[str, Any],
    result_count: int,
    result_file: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> None:
    ResultStore(data_dir).append_log(query, params, result_count, result_file, now=now)
