"""End-to-end integration test for the full pipeline (research → RAG index → insights).

Uses real Google Gemini API for LLM + embeddings and real PostgreSQL+pgvector,
but mocks Brave Search to avoid external search dependency.

Requirements:
  - GOOGLE_API_KEY env var set
  - PostgreSQL+pgvector running on port 5433 (docker compose up -d)

Run:  uv run pytest tests/test_pipeline_e2e.py -v -s -m integration
"""

from __future__ import annotations

import json
import os
import socket

import pytest
from dotenv import load_dotenv

load_dotenv()

PG_URL = "postgresql://postgres:postgres@localhost:5433/rag"
TABLE = "test_pipeline_e2e"


def _pg_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5433), timeout=2):
            return True
    except OSError:
        return False


skip_no_pg = pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable on port 5433")
skip_no_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)

# Counter for generating unique fake URLs
_call_counter = 0


def _fake_brave_search(query: str, count: int = 20, timeout: int = 25, freshness: str | None = None) -> dict:
    """Return fake Brave Web Search API results with 200+ char snippets."""
    global _call_counter
    results = []
    for i in range(min(count, 3)):
        _call_counter += 1
        snippet = (
            f"Result {_call_counter} for '{query}'. "
            "Developers often face significant challenges with slow build times, "
            "poor documentation, and lack of debugging tools. Teams report spending "
            "hours configuring environments and dealing with dependency conflicts. "
            "This is a widespread pain point in the software development industry."
        )
        results.append(
            {
                "url": f"https://example.com/article-{_call_counter}",
                "title": f"Developer Pain Point #{_call_counter}: {query[:40]}",
                "description": snippet,
                "extra_snippets": [
                    "Many teams struggle with CI/CD pipeline complexity and flaky tests "
                    "that slow down release cycles and frustrate engineering teams."
                ],
            }
        )
    return {"web": {"results": results}}


def _drop_table(pg_url: str, table: str) -> None:
    import psycopg

    with psycopg.connect(pg_url) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()


@pytest.mark.integration
@skip_no_pg
@skip_no_google
def test_pipeline_e2e(tmp_path, monkeypatch):
    """Run the full pipeline end-to-end with real Gemini + pgvector, mocked Brave Search."""
    global _call_counter
    _call_counter = 0

    # Mock Brave Search at the module level used by ResearchLoop
    import research_loop.loop as rl_loop

    monkeypatch.setattr(rl_loop, "brave_search", _fake_brave_search)
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token-for-test")

    argv = [
        "--goal", "Find developer tool pain points",
        "--domains", "example.com",
        "--run-id", "test-e2e",
        "--data-dir", str(tmp_path),
        # Research — minimal
        "--research-provider", "gemini",
        "--research-model", "gemini-2.0-flash-lite",
        "--iterations", "1",
        "--per-iter-queries", "2",
        "--results-per-query", "2",
        "--max-docs", "10",
        # RAG
        "--embed-provider", "google",
        "--pg-url", PG_URL,
        "--index", TABLE,
        "--namespace", "test-e2e",
        "--chunk-size", "400",
        "--chunk-overlap", "50",
        # Insights — minimal
        "--insight-provider", "gemini",
        "--insight-model", "gemini-2.0-flash-lite",
        "--insight-iterations", "1",
        "--queries-per-iteration", "2",
        "--top-k", "5",
        "--max-evidence", "10",
    ]

    try:
        # Ensure clean state
        _drop_table(PG_URL, TABLE)

        from pipeline.cli import main

        main(argv)

        run_dir = tmp_path / "test-e2e"

        # 1. params.json exists
        params_path = run_dir / "params.json"
        assert params_path.exists(), "params.json should exist"
        params = json.loads(params_path.read_text())
        assert params["goal"] == "Find developer tool pain points"

        # 2. Corpus JSONL exists and has >= 1 line
        corpus_path = run_dir / "corpus.jsonl"
        assert corpus_path.exists(), "corpus.jsonl should exist"
        corpus_lines = [
            line for line in corpus_path.read_text().splitlines() if line.strip()
        ]
        assert len(corpus_lines) >= 1, f"corpus should have >= 1 doc, got {len(corpus_lines)}"

        # 3. Report markdown exists and has content
        report_path = run_dir / "report.md"
        assert report_path.exists(), "report.md should exist"
        report_text = report_path.read_text()
        assert len(report_text) > 50, f"report should have meaningful content, got {len(report_text)} chars"

        # 4. PostgreSQL table has rows for the test namespace
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {TABLE} WHERE namespace = %s",
                ("test-e2e",),
            ).fetchone()
            assert row[0] > 0, f"Expected rows in DB for namespace 'test-e2e', got {row[0]}"

    finally:
        _drop_table(PG_URL, TABLE)
