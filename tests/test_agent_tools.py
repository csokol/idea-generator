"""Tests for agent.tools — Brave Search LangChain tool wrapper."""

from __future__ import annotations

import httpx
import pytest

from agent.tools import _brave_search, brave_search


FAKE_RESPONSE = {
    "grounding": {
        "generic": [
            {"title": "Result 1", "url": "https://example.com", "snippets": ["snippet one"]},
        ]
    }
}


class FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or FAKE_RESPONSE
        self.text = '{"error": "bad request"}'

    def json(self):
        return self._data


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="BRAVE_SEARCH_TOKEN"):
        _brave_search("test query")


def test_successful_search(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeResponse())
    result = _brave_search("test query")
    assert "Result 1" in result
    assert "example.com" in result


def test_api_error_raises(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeResponse(status_code=429))
    with pytest.raises(RuntimeError, match="HTTP 429"):
        _brave_search("test query")


def test_tool_metadata():
    assert brave_search.name == "brave_search"
    assert "search" in brave_search.description.lower()
