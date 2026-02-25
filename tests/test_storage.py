"""Tests for brave_search storage module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from brave_search.storage import ResultStore, _slugify, append_log, save_result


FIXED_TIME = datetime(2026, 2, 14, 15, 32, 1, tzinfo=timezone.utc)

SAMPLE_DATA = {"grounding": {"generic": [{"title": "A"}, {"title": "B"}]}}


class TestResultStore:
    def test_save_result_creates_file(self, tmp_path: Path) -> None:
        store = ResultStore(data_dir=tmp_path)
        path = store.save_result(SAMPLE_DATA, "python web scraping", now=FIXED_TIME)
        assert path.exists()
        assert path.parent == tmp_path / "results"
        assert path.name == "2026-02-14T15-32-01_python-web-scraping.json"
        parsed = json.loads(path.read_text())
        assert parsed == SAMPLE_DATA

    def test_save_result_creates_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        store = ResultStore(data_dir=nested)
        path = store.save_result(SAMPLE_DATA, "test", now=FIXED_TIME)
        assert path.exists()

    def test_append_log_creates_and_appends(self, tmp_path: Path) -> None:
        store = ResultStore(data_dir=tmp_path)
        result_file = tmp_path / "results" / "file.json"
        params = {"q": "test", "count": 10}

        store.append_log("test", params, 5, result_file, now=FIXED_TIME)
        store.append_log("test2", params, 3, result_file, now=FIXED_TIME)

        log_path = tmp_path / "search.log"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["query"] == "test"
        assert first["count"] == 10
        assert first["result_count"] == 5

        second = json.loads(lines[1])
        assert second["query"] == "test2"


# Shim backward-compatibility tests
def test_save_result_shim(tmp_path: Path) -> None:
    path = save_result(SAMPLE_DATA, "python web scraping", data_dir=tmp_path, now=FIXED_TIME)
    assert path.exists()


def test_append_log_shim(tmp_path: Path) -> None:
    result_file = tmp_path / "results" / "file.json"
    append_log("test", {"count": 10}, 5, result_file, data_dir=tmp_path, now=FIXED_TIME)
    assert (tmp_path / "search.log").exists()


def test_slugify_special_chars() -> None:
    assert _slugify("Hello World!!! @#$") == "hello-world"


def test_slugify_truncation() -> None:
    long_query = "a" * 100
    slug = _slugify(long_query)
    assert len(slug) <= 60
