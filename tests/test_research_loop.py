"""Tests for the research_loop package."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from research_loop.cli import parse_domains
from research_loop.records import (
    Deduplicator,
    append_jsonl,
    build_record,
    canonical_url,
    content_hash,
    url_hash,
)


# ── Domain Parsing ───────────────────────────────────────────────────

class TestParseDomains:
    def test_comma_separated(self):
        assert parse_domains(["a.com,b.com"]) == ["a.com", "b.com"]

    def test_mixed_comma_and_space(self):
        assert parse_domains(["a.com,b.com", "c.com"]) == ["a.com", "b.com", "c.com"]

    def test_space_separated_unchanged(self):
        assert parse_domains(["a.com", "b.com"]) == ["a.com", "b.com"]

    def test_strips_whitespace(self):
        assert parse_domains(["a.com, b.com , c.com"]) == ["a.com", "b.com", "c.com"]

    def test_ignores_empty(self):
        assert parse_domains(["a.com,,b.com"]) == ["a.com", "b.com"]


# ── Records / Dedup ──────────────────────────────────────────────────

class TestCanonicalUrl:
    def test_strips_fragment_and_trailing_slash(self):
        assert canonical_url("https://Example.COM/path/#frag") == "https://example.com/path"

    def test_preserves_query_string(self):
        assert canonical_url("https://example.com/p?a=1") == "https://example.com/p?a=1"

    def test_empty_path_becomes_slash(self):
        assert canonical_url("https://example.com") == "https://example.com/"


class TestHashes:
    def test_url_hash_deterministic(self):
        h1 = url_hash("https://example.com/page")
        h2 = url_hash("https://example.com/page")
        assert h1 == h2
        assert len(h1) == 64

    def test_content_hash_deterministic(self):
        h = content_hash("hello world")
        assert len(h) == 64
        assert h == content_hash("hello world")


class TestDeduplicator:
    def test_new_url_is_not_duplicate(self):
        d = Deduplicator()
        assert not d.is_duplicate("https://example.com/1")

    def test_seen_url_is_duplicate(self):
        d = Deduplicator()
        d.mark_seen("https://example.com/1")
        assert d.is_duplicate("https://example.com/1")

    def test_canonical_dedup(self):
        d = Deduplicator()
        d.mark_seen("https://Example.COM/page/#section")
        assert d.is_duplicate("https://example.com/page")


class TestBuildRecord:
    def test_has_required_fields(self):
        r = build_record(
            run_id="r1", goal="test", domains=["d.com"], iteration=1,
            query="q", rank=0, url="https://d.com/1", title="T",
            snippet="S", source_domain="d.com",
        )
        assert r["run_id"] == "r1"
        assert r["dedupe"] == "unique"
        assert r["errors"] == []
        assert "ts" in r


class TestAppendJsonl:
    def test_writes_parseable_jsonl(self, tmp_path):
        out = tmp_path / "out.jsonl"
        rec1 = {"a": 1, "b": "hello"}
        rec2 = {"a": 2, "b": "world"}
        append_jsonl(rec1, out)
        append_jsonl(rec2, out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == rec1
        assert json.loads(lines[1]) == rec2

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "out.jsonl"
        append_jsonl({"x": 1}, out)
        assert out.exists()


# ── Query Generation ─────────────────────────────────────────────────

class TestQueryGen:
    def test_parse_llm_response(self):
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {"query": "site:example.com test", "why": "reason", "domain": "example.com"}
        ])
        mock_llm.invoke.return_value = mock_response

        gen = QueryGenerator(mock_llm, goal="test goal", domains=["example.com"])
        result = gen.generate(num_queries=1)
        assert len(result) == 1
        assert result[0]["query"] == "site:example.com test"

    def test_research_summary_in_prompt(self):
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {"query": "site:example.com deep dive", "why": "reason", "domain": "example.com"}
        ])
        mock_llm.invoke.return_value = mock_response

        gen = QueryGenerator(mock_llm, goal="test goal", domains=["example.com"])
        gen.generate(num_queries=1, research_summary="Users complain about slow loading times")

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Users complain about slow loading times" in prompt_sent

    def test_fallback_on_invalid_json(self):
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "this is not json at all"
        mock_llm.invoke.return_value = mock_response

        gen = QueryGenerator(mock_llm, goal="find apps", domains=["example.com", "other.com"])
        result = gen.generate(num_queries=3)
        assert len(result) == 3
        assert all("site:" in q["query"] for q in result)


class TestSummarizeFindings:
    def test_returns_summary(self):
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Key themes: users want faster checkout."
        mock_llm.invoke.return_value = mock_response

        gen = QueryGenerator(mock_llm, goal="find pain points", domains=["example.com"])
        results = [
            {"title": "Slow checkout", "snippet": "Users report slow checkout", "url": "https://example.com/1"},
            {"title": "Cart issues", "snippet": "Cart abandonment is high", "url": "https://example.com/2"},
        ]
        summary = gen.summarize_findings(results)
        assert summary == "Key themes: users want faster checkout."

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Slow checkout" in prompt_sent
        assert "Cart issues" in prompt_sent
        assert "find pain points" in prompt_sent

    def test_empty_results(self):
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        gen = QueryGenerator(mock_llm, goal="test", domains=["example.com"])
        assert gen.summarize_findings([]) == ""
        mock_llm.invoke.assert_not_called()

    def test_list_content_from_gemini(self):
        """Gemini sometimes returns response.content as a list of parts."""
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        # Simulate Gemini returning content as a list of text blocks
        mock_response.content = [{"text": "Key themes: users want better tools."}]
        mock_llm.invoke.return_value = mock_response

        gen = QueryGenerator(mock_llm, goal="find pain points", domains=["example.com"])
        results = [{"title": "Test", "snippet": "test snippet", "url": "https://example.com/1"}]
        summary = gen.summarize_findings(results)
        assert summary == "Key themes: users want better tools."

    def test_generate_handles_list_content(self):
        """Query generation should handle list content from Gemini."""
        from research_loop.query_gen import QueryGenerator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [json.dumps([
            {"query": "site:example.com test", "why": "reason", "domain": "example.com"}
        ])]
        mock_llm.invoke.return_value = mock_response

        gen = QueryGenerator(mock_llm, goal="test", domains=["example.com"])
        result = gen.generate(num_queries=1)
        assert len(result) == 1
        assert result[0]["query"] == "site:example.com test"


# ── Search Wrapper ───────────────────────────────────────────────────

FAKE_SEARCH_DATA = {
    "web": {
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "description": "s1", "extra_snippets": []},
            {"title": "Result 2", "url": "https://example.com/2", "description": "s2", "extra_snippets": []},
        ]
    }
}


class TestSearch:
    def test_success(self, monkeypatch):
        from research_loop.search import BraveSearchClient

        def mock_get(*a, **kw):
            return httpx.Response(200, json=FAKE_SEARCH_DATA)

        monkeypatch.setattr(httpx, "get", mock_get)

        client = BraveSearchClient(token="tok")
        data = client.search("test query")
        results = client.extract_results(data)
        assert len(results) == 2
        assert results[0]["title"] == "Result 1"

    def test_missing_token(self):
        from research_loop.search import BraveSearchClient

        client = BraveSearchClient(token="")
        with pytest.raises(RuntimeError, match="BRAVE_SEARCH_TOKEN"):
            client.search("query")

    def test_http_error(self, monkeypatch):
        from research_loop.search import BraveSearchClient

        def mock_get(*a, **kw):
            return httpx.Response(429, text="rate limited")

        monkeypatch.setattr(httpx, "get", mock_get)
        client = BraveSearchClient(token="tok")
        with pytest.raises(RuntimeError, match="429"):
            client.search("query")

    def test_freshness_param_included(self, monkeypatch):
        from research_loop.search import BraveSearchClient

        captured_params = {}

        def mock_get(*a, **kw):
            captured_params.update(kw.get("params", {}))
            return httpx.Response(200, json=FAKE_SEARCH_DATA)

        monkeypatch.setattr(httpx, "get", mock_get)
        client = BraveSearchClient(token="tok")
        client.search("test", freshness="pw")
        assert captured_params["freshness"] == "pw"

    def test_freshness_param_omitted_when_none(self, monkeypatch):
        from research_loop.search import BraveSearchClient

        captured_params = {}

        def mock_get(*a, **kw):
            captured_params.update(kw.get("params", {}))
            return httpx.Response(200, json=FAKE_SEARCH_DATA)

        monkeypatch.setattr(httpx, "get", mock_get)
        client = BraveSearchClient(token="tok")
        client.search("test")
        assert "freshness" not in captured_params


# ── Fetch ────────────────────────────────────────────────────────────

class TestFetch:
    def test_extracts_text_strips_scripts(self, monkeypatch):
        from research_loop import fetch

        html = "<html><head><script>evil()</script><style>body{}</style></head><body><p>Hello world</p></body></html>"

        def mock_get(*a, **kw):
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})

        monkeypatch.setattr(httpx, "get", mock_get)
        result = fetch.fetch_page("https://example.com")
        assert result["fetched"] is True
        assert "Hello world" in result["text"]
        assert "evil" not in result["text"]

    def test_handles_error(self, monkeypatch):
        from research_loop import fetch

        def mock_get(*a, **kw):
            raise httpx.ConnectError("fail")

        monkeypatch.setattr(httpx, "get", mock_get)
        result = fetch.fetch_page("https://unreachable.test")
        assert result["fetched"] is False
        assert result["text"] is None


# ── Loop Integration ─────────────────────────────────────────────────

def _make_mock_llm():
    """Create a mock LLM that handles both query gen and summarize calls."""
    mock_llm = MagicMock()
    query_response = MagicMock()
    query_response.content = json.dumps([
        {"query": "site:example.com test", "why": "r", "domain": "example.com"}
    ])
    summary_response = MagicMock()
    summary_response.content = "Summary of findings."

    def side_effect(prompt):
        if "Summarize the findings" in prompt:
            return summary_response
        return query_response

    mock_llm.invoke.side_effect = side_effect
    return mock_llm


class TestLoop:
    def test_max_docs_stopping(self, monkeypatch, tmp_path):
        from research_loop import loop
        from research_loop.loop import ResearchLoop

        monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "tok")
        monkeypatch.setenv("GROQ_API_KEY", "key")

        mock_llm = _make_mock_llm()
        monkeypatch.setattr(loop, "build_llm", lambda *a, **kw: mock_llm)

        def mock_search(query, count=20, timeout=25, freshness=None):
            return {
                "web": {
                    "results": [
                        {"title": f"R-{i}", "url": f"https://example.com/{query}/{i}", "description": f"s{i}", "extra_snippets": []}
                        for i in range(count)
                    ]
                }
            }

        monkeypatch.setattr(loop, "brave_search", mock_search)

        out = tmp_path / "test.jsonl"
        rl = ResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=5,
            per_iter_queries=5,
            results_per_query=5,
            max_docs=3,
            out=str(out),
            sleep_sec=0,
        )
        rl.run()

        lines = out.read_text().strip().split("\n")
        unique = [json.loads(l) for l in lines if json.loads(l)["dedupe"] == "unique"]
        assert len(unique) == 3

    def test_dedup_skipping(self, monkeypatch, tmp_path):
        from research_loop import loop
        from research_loop.loop import ResearchLoop

        monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "tok")
        monkeypatch.setenv("GROQ_API_KEY", "key")

        mock_llm = MagicMock()
        query_response = MagicMock()
        query_response.content = json.dumps([
            {"query": "site:example.com q1", "why": "r", "domain": "example.com"},
            {"query": "site:example.com q2", "why": "r", "domain": "example.com"},
        ])
        summary_response = MagicMock()
        summary_response.content = "Summary."

        def side_effect(prompt):
            if "Summarize the findings" in prompt:
                return summary_response
            return query_response

        mock_llm.invoke.side_effect = side_effect
        monkeypatch.setattr(loop, "build_llm", lambda *a, **kw: mock_llm)

        def mock_search(query, count=20, timeout=25, freshness=None):
            # Always return the same URL
            return {
                "web": {
                    "results": [
                        {"title": "Same", "url": "https://example.com/same", "description": "s", "extra_snippets": []}
                    ]
                }
            }

        monkeypatch.setattr(loop, "brave_search", mock_search)

        out = tmp_path / "dedup.jsonl"
        rl = ResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=1,
            per_iter_queries=2,
            results_per_query=1,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        rl.run()

        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        unique = [l for l in lines if l["dedupe"] == "unique"]
        dupes = [l for l in lines if l["dedupe"] == "duplicate"]
        assert len(unique) == 1
        assert len(dupes) == 1

    def test_dry_run_skips_fetch(self, monkeypatch, tmp_path):
        from research_loop import fetch, loop
        from research_loop.loop import ResearchLoop

        monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "tok")
        monkeypatch.setenv("GROQ_API_KEY", "key")

        mock_llm = _make_mock_llm()
        monkeypatch.setattr(loop, "build_llm", lambda *a, **kw: mock_llm)

        def mock_search(query, count=20, timeout=25, freshness=None):
            return {
                "web": {
                    "results": [{"title": "R", "url": "https://example.com/r", "description": "s", "extra_snippets": []}]
                }
            }

        monkeypatch.setattr(loop, "brave_search", mock_search)

        fetch_called = []
        original_fetch = fetch.fetch_page

        def spy_fetch(*a, **kw):
            fetch_called.append(True)
            return original_fetch(*a, **kw)

        monkeypatch.setattr(fetch, "fetch_page", spy_fetch)

        out = tmp_path / "dry.jsonl"
        rl = ResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=1,
            per_iter_queries=1,
            results_per_query=1,
            max_docs=10,
            out=str(out),
            fetch_pages=True,
            dry_run=True,
            sleep_sec=0,
        )
        rl.run()

        assert len(fetch_called) == 0
        assert out.exists()

    def test_summaries_accumulate_across_iterations(self, monkeypatch, tmp_path):
        from research_loop import loop
        from research_loop.loop import ResearchLoop

        monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "tok")
        monkeypatch.setenv("GROQ_API_KEY", "key")

        call_count = {"summarize": 0, "generate": 0}

        mock_llm = MagicMock()
        query_response = MagicMock()
        query_response.content = json.dumps([
            {"query": "site:example.com test", "why": "r", "domain": "example.com"}
        ])
        summary_response = MagicMock()
        summary_response.content = "Iteration summary."

        slug_response = MagicMock()
        slug_response.content = "test-slug"

        def side_effect(prompt):
            if "filesystem-safe slug" in prompt:
                return slug_response
            if "Summarize the findings" in prompt:
                call_count["summarize"] += 1
                return summary_response
            call_count["generate"] += 1
            # On iteration 2+, the prompt should contain prior summaries
            if call_count["generate"] > 1:
                assert "Iteration summary." in prompt
            return query_response

        mock_llm.invoke.side_effect = side_effect
        monkeypatch.setattr(loop, "build_llm", lambda *a, **kw: mock_llm)

        call_idx = {"n": 0}
        def mock_search(query, count=20, timeout=25, freshness=None):
            call_idx["n"] += 1
            return {
                "web": {
                    "results": [
                        {"title": f"R-{call_idx['n']}", "url": f"https://example.com/{call_idx['n']}", "description": f"s{call_idx['n']}", "extra_snippets": []}
                    ]
                }
            }

        monkeypatch.setattr(loop, "brave_search", mock_search)

        out = tmp_path / "summaries.jsonl"
        rl = ResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=3,
            per_iter_queries=1,
            results_per_query=1,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        rl.run()

        assert call_count["summarize"] == 3
        assert len(rl.iteration_summaries) == 3

    def test_comma_separated_domains_via_cli(self, monkeypatch, tmp_path):
        """Verify --domains a.com,b.com is properly split into ['a.com', 'b.com']."""
        from research_loop import loop
        from research_loop.loop import ResearchLoop

        monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "tok")
        monkeypatch.setenv("GROQ_API_KEY", "key")

        mock_llm = MagicMock()
        query_response = MagicMock()
        query_response.content = json.dumps([
            {"query": "site:a.com test", "why": "r", "domain": "a.com"},
            {"query": "site:b.com test", "why": "r", "domain": "b.com"},
        ])
        summary_response = MagicMock()
        summary_response.content = "Summary."
        slug_response = MagicMock()
        slug_response.content = "test"

        def side_effect(prompt):
            if "filesystem-safe slug" in prompt:
                return slug_response
            if "Summarize the findings" in prompt:
                return summary_response
            return query_response

        mock_llm.invoke.side_effect = side_effect
        monkeypatch.setattr(loop, "build_llm", lambda *a, **kw: mock_llm)

        call_idx = {"n": 0}
        def mock_search(query, count=20, timeout=25, freshness=None):
            call_idx["n"] += 1
            return {
                "web": {
                    "results": [
                        {"title": f"R-{call_idx['n']}", "url": f"https://example.com/{call_idx['n']}", "description": "snippet", "extra_snippets": []}
                    ]
                }
            }

        monkeypatch.setattr(loop, "brave_search", mock_search)

        out = tmp_path / "comma-domains.jsonl"

        from research_loop.cli import main as research_main

        research_main([
            "--goal", "test goal",
            "--domains", "a.com,b.com",
            "--iterations", "1",
            "--per-iter-queries", "2",
            "--results-per-query", "1",
            "--max-docs", "10",
            "--out", str(out),
            "--dry-run",
        ])

        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        # Verify domains field in records contains the properly split list
        for record in lines:
            assert record["domains"] == ["a.com", "b.com"], (
                f"Expected ['a.com', 'b.com'], got {record['domains']}"
            )
