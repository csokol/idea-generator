"""Tests for brave_search CLI."""

from __future__ import annotations

import json

import httpx
import pytest

from brave_search.cli import format_output, main, parse_args


def _mock_response(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=data)


FAKE_RESULTS = {
    "grounding": {
        "generic": [
            {"title": "First Result", "url": "https://example.com/1", "snippets": ["Snippet one", "Extra detail"]},
            {"title": "Second Result", "url": "https://example.com/2", "snippets": ["Snippet two"]},
        ]
    },
    "sources": {
        "https://example.com/1": {"name": "Example 1"},
        "https://example.com/2": {"name": "Example 2"},
    },
}


def test_missing_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_TOKEN", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    with pytest.raises(SystemExit) as exc:
        main(["hello"])
    assert exc.value.code == 2


def test_format_human() -> None:
    output = format_output(FAKE_RESULTS, mode="human")
    assert "1. First Result" in output
    assert "https://example.com/1" in output
    assert "Snippet one" in output
    assert "Extra detail" in output
    assert "2. Second Result" in output


def test_format_json() -> None:
    output = format_output(FAKE_RESULTS, mode="json")
    items = json.loads(output)
    assert len(items) == 2
    assert items[0]["title"] == "First Result"
    assert items[0]["snippets"] == ["Snippet one", "Extra detail"]
    assert items[1]["url"] == "https://example.com/2"


def test_format_raw() -> None:
    output = format_output(FAKE_RESULTS, mode="raw")
    parsed = json.loads(output)
    assert parsed == FAKE_RESULTS


def test_format_empty_results() -> None:
    assert format_output({"grounding": {"generic": []}}, mode="human") == "No results found."
    assert format_output({}, mode="human") == "No results found."


def test_successful_search(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "test-token")

    def mock_get(*args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json=FAKE_RESULTS)

    monkeypatch.setattr(httpx, "get", mock_get)
    main(["test query", "--no-save"])
    captured = capsys.readouterr()
    assert "First Result" in captured.out


def test_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "test-token")

    def mock_get(*args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    monkeypatch.setattr(httpx, "get", mock_get)
    with pytest.raises(SystemExit) as exc:
        main(["test query", "--no-save"])
    assert exc.value.code == 3


def test_parse_args_defaults() -> None:
    args = parse_args(["my query"])
    assert args.query == "my query"
    assert args.count == 20
    assert args.max_urls is None
    assert args.max_tokens is None
    assert args.max_snippets is None
    assert args.threshold_mode is None
    assert not args.json_mode
    assert not args.raw
