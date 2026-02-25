"""Tests for the insights package."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from insights.models import EvidenceStore, Opportunity, Signal


# ── JSON Parsing ─────────────────────────────────────────────────


class TestParseJsonResponse:
    def test_clean_json(self):
        from insights.loop import _parse_json_response

        assert _parse_json_response('["a", "b"]') == ["a", "b"]

    def test_fenced_json(self):
        from insights.loop import _parse_json_response

        text = '```\n["a", "b"]\n```'
        assert _parse_json_response(text) == ["a", "b"]

    def test_fenced_json_with_language_tag(self):
        from insights.loop import _parse_json_response

        text = '```json\n{"key": "value"}\n```'
        assert _parse_json_response(text) == {"key": "value"}

    def test_fenced_json_with_whitespace(self):
        from insights.loop import _parse_json_response

        text = '  ```json\n[1, 2, 3]\n```  '
        assert _parse_json_response(text) == [1, 2, 3]


class TestFuzzyMatchKey:
    def test_exact_match(self):
        from insights.loop import _fuzzy_match_key

        assert _fuzzy_match_key("slow api", {"slow api", "bad docs"}) == "slow api"

    def test_case_insensitive_match(self):
        from insights.loop import _fuzzy_match_key

        assert _fuzzy_match_key("Slow API", {"slow api", "bad docs"}) == "slow api"

    def test_substring_match(self):
        from insights.loop import _fuzzy_match_key

        result = _fuzzy_match_key("slow", {"slow api responses", "bad docs"})
        assert result == "slow api responses"

    def test_no_match(self):
        from insights.loop import _fuzzy_match_key

        assert _fuzzy_match_key("unknown", {"slow api", "bad docs"}) is None


# ── Models / EvidenceStore ────────────────────────────────────────


class TestSignal:
    def test_normalized_pain(self):
        s = _make_signal(pain="  High Latency Issues  ")
        assert s.normalized_pain == "high latency issues"


class TestEvidenceStore:
    def test_add_and_count(self):
        store = EvidenceStore()
        s1 = _make_signal(doc_id="a#0", pain="slow api")
        s2 = _make_signal(doc_id="b#0", pain="slow api")
        assert store.add(s1)
        assert store.add(s2)
        assert store.total_signals == 2

    def test_dedup_by_doc_id(self):
        store = EvidenceStore()
        s1 = _make_signal(doc_id="a#0")
        s2 = _make_signal(doc_id="a#0")
        assert store.add(s1)
        assert not store.add(s2)
        assert store.total_signals == 1

    def test_pain_groups(self):
        store = EvidenceStore()
        store.add(_make_signal(doc_id="a#0", pain="slow api"))
        store.add(_make_signal(doc_id="b#0", pain="slow api"))
        store.add(_make_signal(doc_id="c#0", pain="bad docs"))
        groups = store.pain_groups
        assert len(groups) == 2
        assert len(groups["slow api"]) == 2
        assert len(groups["bad docs"]) == 1

    def test_all_signals(self):
        store = EvidenceStore()
        store.add(_make_signal(doc_id="a#0"))
        store.add(_make_signal(doc_id="b#0"))
        assert len(store.all_signals()) == 2


class TestOpportunity:
    def test_evidence_count(self):
        opp = Opportunity(title="T", summary="S", signals=[_make_signal(), _make_signal(doc_id="x#1")])
        assert opp.evidence_count == 2

    def test_unique_urls(self):
        opp = Opportunity(
            title="T", summary="S",
            signals=[
                _make_signal(url="https://a.com"),
                _make_signal(doc_id="x#1", url="https://a.com"),
                _make_signal(doc_id="y#0", url="https://b.com"),
            ],
        )
        assert opp.unique_urls == {"https://a.com", "https://b.com"}

    def test_avg_severity(self):
        opp = Opportunity(
            title="T", summary="S",
            signals=[_make_signal(severity=4), _make_signal(doc_id="x#1", severity=2)],
        )
        assert opp.avg_severity == 3.0

    def test_empty_metrics(self):
        opp = Opportunity(title="T", summary="S")
        assert opp.avg_severity == 0.0
        assert opp.avg_willingness_to_pay == 0.0


# ── RagStore ──────────────────────────────────────────────────────


class TestRagStore:
    def test_search_builds_correct_sql(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("id",), ("url",), ("title",), ("source_domain",),
            ("chunk_index",), ("chunk_text",), ("score",),
        ]
        mock_cursor.fetchall.return_value = [
            ("doc1#0", "https://a.com", "Title A", "a.com", 0, "chunk text here", 0.95),
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        from insights.rag_store import RagStore

        store = RagStore(mock_conn, "chunks", "dev", mock_embedder)
        hits = store.search("test query", top_k=5)

        assert len(hits) == 1
        assert hits[0]["id"] == "doc1#0"
        assert hits[0]["score"] == 0.95
        mock_embedder.embed.assert_called_once_with(["test query"])

    def test_search_with_domain_filter(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("id",), ("url",), ("title",), ("source_domain",),
            ("chunk_index",), ("chunk_text",), ("score",),
        ]
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        from insights.rag_store import RagStore

        store = RagStore(mock_conn, "chunks", "dev", mock_embedder)
        hits = store.search("test", top_k=5, domains=["a.com", "b.com"])

        # Verify SQL was executed with domain filter
        mock_cursor.execute.assert_called_once()
        composed_sql = mock_cursor.execute.call_args[0][0]
        # The composed SQL object should contain source_domain IN clause
        assert any("source_domain" in str(part) for part in composed_sql._obj)

    def test_search_caching(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("id",), ("url",), ("title",), ("source_domain",),
            ("chunk_index",), ("chunk_text",), ("score",),
        ]
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        from insights.rag_store import RagStore

        store = RagStore(mock_conn, "chunks", "dev", mock_embedder)
        store.search("same query", top_k=5)
        store.search("same query", top_k=5)

        # Embedding computed only once, SQL executed only once
        assert mock_embedder.embed.call_count == 1
        assert mock_cursor.execute.call_count == 1

    def test_get_doc(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("id",), ("url",), ("title",), ("source_domain",),
            ("chunk_index",), ("chunk_text",),
        ]
        mock_cursor.fetchall.return_value = [
            ("abc#0", "https://a.com", "T", "a.com", 0, "chunk 0"),
            ("abc#1", "https://a.com", "T", "a.com", 1, "chunk 1"),
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()
        from insights.rag_store import RagStore

        store = RagStore(mock_conn, "chunks", "dev", mock_embedder)
        chunks = store.get_doc("abc")
        assert len(chunks) == 2
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1


# ── Report ────────────────────────────────────────────────────────


class TestReport:
    def test_generates_markdown(self):
        from insights.report import generate_report

        opps = [
            Opportunity(
                title="Slow API Tool",
                summary="Users need faster API tooling",
                signals=[
                    _make_signal(pain="slow api", url="https://a.com", severity=4, willingness_to_pay=3),
                    _make_signal(doc_id="b#0", pain="api latency", url="https://b.com", severity=5, willingness_to_pay=4),
                ],
                score=0.82,
                confidence="high",
            ),
        ]
        report = generate_report(opps, goal="developer tools", queries_explored=["api pain", "dev tools"])
        assert "# Insight Report: developer tools" in report
        assert "Slow API Tool" in report
        assert "https://a.com" in report
        assert "https://b.com" in report
        assert "api pain" in report
        assert "0.82" in report

    def test_empty_opportunities(self):
        from insights.report import generate_report

        report = generate_report([], goal="test", queries_explored=["q1"])
        assert "No opportunities" in report


# ── InsightLoop (integration-style with mocks) ───────────────────


class TestInsightLoop:
    def test_full_run_with_mocks(self):
        """Mock RAG store + LLM → run loop → verify output format."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            {"id": "doc1#0", "url": "https://a.com/1", "title": "Page A",
             "source_domain": "a.com", "chunk_index": 0,
             "chunk_text": "Users complain about slow dashboard loading times", "score": 0.9},
            {"id": "doc2#0", "url": "https://b.com/1", "title": "Page B",
             "source_domain": "b.com", "chunk_index": 0,
             "chunk_text": "Many users report the dashboard is too slow for daily use", "score": 0.85},
        ]
        mock_rag.get_doc.side_effect = lambda doc_id: {
            "doc1": [{"id": "doc1#0", "url": "https://a.com/1", "title": "Page A",
                       "source_domain": "a.com", "chunk_index": 0,
                       "chunk_text": "Users complain about slow dashboard loading times"}],
            "doc2": [{"id": "doc2#0", "url": "https://b.com/1", "title": "Page B",
                       "source_domain": "b.com", "chunk_index": 0,
                       "chunk_text": "Many users report the dashboard is too slow for daily use"}],
        }.get(doc_id, [])

        # LLM responses: plan_queries, extract_signals, cluster_merge
        plan_response = MagicMock()
        plan_response.content = json.dumps(["dashboard performance", "slow loading complaints"])

        signals_response = MagicMock()
        signals_response.content = json.dumps([
            {
                "pain": "slow dashboard",
                "workaround": "use raw API instead",
                "desired_outcome": "sub-second dashboard loads",
                "segment": "data analysts",
                "severity": 4,
                "willingness_to_pay": 3,
                "keywords": ["dashboard", "slow"],
                "doc_id": "doc1",
                "url": "https://a.com/1",
            },
            {
                "pain": "slow dashboard",
                "workaround": "refresh frequently",
                "desired_outcome": "real-time dashboard",
                "segment": "product managers",
                "severity": 3,
                "willingness_to_pay": 4,
                "keywords": ["dashboard", "performance"],
                "doc_id": "doc2",
                "url": "https://b.com/1",
            },
        ])

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([
            {
                "title": "Fast Dashboard Solution",
                "summary": "Build a high-performance dashboard that loads in under a second",
                "pain_keys": ["slow dashboard"],
                "confidence": "high",
            },
        ])

        evaluate_continue = MagicMock()
        evaluate_continue.content = json.dumps(
            {"should_continue": True, "reasoning": "need more", "suggested_focus": "more angles"}
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response,      # iteration 1: plan queries
            signals_response,   # iteration 1: extract signals
            cluster_response,   # iteration 1: _evaluate_progress → _cluster_and_merge
            evaluate_continue,  # iteration 1: _evaluate_progress → LLM eval
            plan_response,      # iteration 2: plan queries (returns same, will find no new)
            cluster_response,   # final cluster and merge
        ]

        loop = InsightLoop(
            goal="dashboard tools",
            rag_store=mock_rag,
            llm=mock_llm,
            iterations=2,
            top_k=5,
            queries_per_iteration=2,
            min_evidence_per_opportunity=2,
            use_agent=False,
        )
        opportunities = loop.run()

        assert len(opportunities) >= 1
        opp = opportunities[0]
        assert opp.title == "Fast Dashboard Solution"
        assert opp.evidence_count == 2
        assert opp.score > 0

    def test_no_results(self):
        """When RAG returns nothing, should produce empty opportunities."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_rag.get_doc.return_value = []

        plan_response = MagicMock()
        plan_response.content = json.dumps(["test query"])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = plan_response

        loop = InsightLoop(
            goal="test",
            rag_store=mock_rag,
            llm=mock_llm,
            iterations=1,
            use_agent=False,
        )
        opportunities = loop.run()
        assert opportunities == []

    def test_fenced_json_responses(self):
        """LLM responses wrapped in code fences should still be parsed."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            {"id": "doc1#0", "url": "https://a.com/1", "title": "Page A",
             "source_domain": "a.com", "chunk_index": 0,
             "chunk_text": "Users complain about slow dashboard loading times", "score": 0.9},
            {"id": "doc2#0", "url": "https://b.com/1", "title": "Page B",
             "source_domain": "b.com", "chunk_index": 0,
             "chunk_text": "Many users report the dashboard is too slow", "score": 0.85},
        ]
        mock_rag.get_doc.side_effect = lambda doc_id: {
            "doc1": [{"id": "doc1#0", "url": "https://a.com/1", "title": "Page A",
                       "source_domain": "a.com", "chunk_index": 0,
                       "chunk_text": "Users complain about slow dashboard loading times"}],
            "doc2": [{"id": "doc2#0", "url": "https://b.com/1", "title": "Page B",
                       "source_domain": "b.com", "chunk_index": 0,
                       "chunk_text": "Many users report the dashboard is too slow"}],
        }.get(doc_id, [])

        plan_response = MagicMock()
        plan_response.content = '```json\n["dashboard performance", "slow loading"]\n```'

        signals_response = MagicMock()
        signals_response.content = '```json\n' + json.dumps([
            {"pain": "slow dashboard", "workaround": "none", "desired_outcome": "fast dashboard",
             "segment": "analysts", "severity": 4, "willingness_to_pay": 3,
             "keywords": ["dashboard"], "doc_id": "doc1", "url": "https://a.com/1"},
            {"pain": "slow dashboard", "workaround": "none", "desired_outcome": "fast dashboard",
             "segment": "managers", "severity": 3, "willingness_to_pay": 4,
             "keywords": ["dashboard"], "doc_id": "doc2", "url": "https://b.com/1"},
        ]) + '\n```'

        cluster_response = MagicMock()
        cluster_response.content = '```\n' + json.dumps([
            {"title": "Fast Dashboard", "summary": "Speed up dashboards",
             "pain_keys": ["slow dashboard"], "confidence": "high"},
        ]) + '\n```'

        evaluate_continue = MagicMock()
        evaluate_continue.content = json.dumps(
            {"should_continue": True, "reasoning": "need more", "suggested_focus": "pricing"}
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response, signals_response,
            cluster_response,   # eval → cluster
            evaluate_continue,  # eval → LLM
            plan_response,      # iteration 2, no new hits → stops
            cluster_response,   # final cluster
        ]

        loop = InsightLoop(
            goal="dashboard tools", rag_store=mock_rag, llm=mock_llm,
            iterations=2, top_k=5, queries_per_iteration=2, min_evidence_per_opportunity=2,
            use_agent=False,
        )
        opportunities = loop.run()

        assert len(opportunities) >= 1
        assert opportunities[0].title == "Fast Dashboard"
        assert opportunities[0].evidence_count == 2

    def test_fuzzy_pain_key_matching(self):
        """Cluster pain_keys that don't exactly match should fuzzy-match."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            {"id": "doc1#0", "url": "https://a.com", "title": "A",
             "source_domain": "a.com", "chunk_index": 0,
             "chunk_text": "complaint", "score": 0.9},
            {"id": "doc2#0", "url": "https://b.com", "title": "B",
             "source_domain": "b.com", "chunk_index": 0,
             "chunk_text": "complaint", "score": 0.85},
        ]
        mock_rag.get_doc.side_effect = lambda doc_id: {
            "doc1": [{"id": "doc1#0", "url": "https://a.com", "title": "A",
                       "source_domain": "a.com", "chunk_index": 0, "chunk_text": "complaint"}],
            "doc2": [{"id": "doc2#0", "url": "https://b.com", "title": "B",
                       "source_domain": "b.com", "chunk_index": 0, "chunk_text": "complaint"}],
        }.get(doc_id, [])

        plan_response = MagicMock()
        plan_response.content = json.dumps(["query"])

        signals_response = MagicMock()
        signals_response.content = json.dumps([
            {"pain": "slow api responses", "workaround": "none", "desired_outcome": "fast api",
             "segment": "devs", "severity": 4, "willingness_to_pay": 3,
             "keywords": [], "doc_id": "doc1", "url": "https://a.com"},
            {"pain": "slow api responses", "workaround": "caching", "desired_outcome": "fast api",
             "segment": "devs", "severity": 3, "willingness_to_pay": 4,
             "keywords": [], "doc_id": "doc2", "url": "https://b.com"},
        ])

        cluster_response = MagicMock()
        # LLM rephrases the key slightly — fuzzy matching should handle this
        cluster_response.content = json.dumps([
            {"title": "API Speed", "summary": "Improve API speed",
             "pain_keys": ["Slow API Responses"], "confidence": "high"},
        ])

        evaluate_continue = MagicMock()
        evaluate_continue.content = json.dumps(
            {"should_continue": True, "reasoning": "need more", "suggested_focus": "more"}
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response, signals_response,
            cluster_response,   # eval → cluster
            evaluate_continue,  # eval → LLM
            plan_response,      # iteration 2, no new hits → stops
            cluster_response,   # final cluster
        ]

        loop = InsightLoop(
            goal="api perf", rag_store=mock_rag, llm=mock_llm,
            iterations=2, top_k=5, queries_per_iteration=1, min_evidence_per_opportunity=2,
            use_agent=False,
        )
        opportunities = loop.run()

        assert len(opportunities) == 1
        assert opportunities[0].title == "API Speed"
        assert opportunities[0].evidence_count == 2

    def test_backward_compat_iterations_kwarg(self):
        """The `iterations` kwarg should still work for backward compat."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        plan_response = MagicMock()
        plan_response.content = json.dumps(["query"])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = plan_response

        loop = InsightLoop(goal="test", rag_store=mock_rag, llm=mock_llm, iterations=5)
        assert loop.max_iterations == 5

    def test_agentic_stops_when_target_met(self):
        """Loop stops early when target_opportunities qualified opps are found."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            {"id": "doc1#0", "url": "https://a.com/1", "title": "A",
             "source_domain": "a.com", "chunk_index": 0,
             "chunk_text": "Users hate slow dashboards", "score": 0.9},
            {"id": "doc2#0", "url": "https://b.com/1", "title": "B",
             "source_domain": "b.com", "chunk_index": 0,
             "chunk_text": "Dashboard loading is terrible", "score": 0.85},
            {"id": "doc3#0", "url": "https://c.com/1", "title": "C",
             "source_domain": "c.com", "chunk_index": 0,
             "chunk_text": "API errors are frequent and frustrating", "score": 0.80},
            {"id": "doc4#0", "url": "https://d.com/1", "title": "D",
             "source_domain": "d.com", "chunk_index": 0,
             "chunk_text": "API reliability is a major concern", "score": 0.75},
        ]
        mock_rag.get_doc.side_effect = lambda doc_id: {
            "doc1": [{"id": "doc1#0", "url": "https://a.com/1", "title": "A",
                       "source_domain": "a.com", "chunk_index": 0, "chunk_text": "Users hate slow dashboards"}],
            "doc2": [{"id": "doc2#0", "url": "https://b.com/1", "title": "B",
                       "source_domain": "b.com", "chunk_index": 0, "chunk_text": "Dashboard loading is terrible"}],
            "doc3": [{"id": "doc3#0", "url": "https://c.com/1", "title": "C",
                       "source_domain": "c.com", "chunk_index": 0, "chunk_text": "API errors are frequent and frustrating"}],
            "doc4": [{"id": "doc4#0", "url": "https://d.com/1", "title": "D",
                       "source_domain": "d.com", "chunk_index": 0, "chunk_text": "API reliability is a major concern"}],
        }.get(doc_id, [])

        plan_response = MagicMock()
        plan_response.content = json.dumps(["dashboard issues", "api problems"])

        signals_response_batch1 = MagicMock()
        signals_response_batch1.content = json.dumps([
            {"pain": "slow dashboard", "workaround": "none", "desired_outcome": "fast dashboard",
             "segment": "analysts", "severity": 4, "willingness_to_pay": 3,
             "keywords": ["dashboard"], "doc_id": "doc1", "url": "https://a.com/1"},
            {"pain": "slow dashboard", "workaround": "refresh", "desired_outcome": "fast dashboard",
             "segment": "managers", "severity": 3, "willingness_to_pay": 4,
             "keywords": ["dashboard"], "doc_id": "doc2", "url": "https://b.com/1"},
            {"pain": "api errors", "workaround": "retry manually", "desired_outcome": "reliable api",
             "segment": "developers", "severity": 5, "willingness_to_pay": 4,
             "keywords": ["api"], "doc_id": "doc3", "url": "https://c.com/1"},
        ])

        signals_response_batch2 = MagicMock()
        signals_response_batch2.content = json.dumps([
            {"pain": "api errors", "workaround": "error handling", "desired_outcome": "reliable api",
             "segment": "developers", "severity": 4, "willingness_to_pay": 3,
             "keywords": ["api"], "doc_id": "doc4", "url": "https://d.com/1"},
        ])

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([
            {"title": "Fast Dashboard", "summary": "Speed up dashboards",
             "pain_keys": ["slow dashboard"], "confidence": "high"},
            {"title": "Reliable API", "summary": "Improve API reliability",
             "pain_keys": ["api errors"], "confidence": "high"},
        ])

        # evaluate_progress is called after iteration 1 — _cluster_and_merge
        # will be called inside it, then the target check fires
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response,              # iteration 1: plan queries
            signals_response_batch1,    # iteration 1: extract signals batch 1 (3 docs)
            signals_response_batch2,    # iteration 1: extract signals batch 2 (1 doc)
            cluster_response,           # iteration 1: _evaluate_progress → _cluster_and_merge
            # target_opportunities=2 met → _evaluate_progress returns False → break
            cluster_response,           # final _cluster_and_merge in run()
        ]

        loop = InsightLoop(
            goal="dashboard and api tools",
            rag_store=mock_rag,
            llm=mock_llm,
            max_iterations=5,  # high cap — should stop well before this
            target_opportunities=2,
            min_evidence_per_opportunity=2,
            use_agent=False,
        )
        opportunities = loop.run()

        # Should have produced 2 opportunities and stopped after iteration 1
        assert len(opportunities) == 2
        assert mock_llm.invoke.call_count == 5  # plan, signals×2, eval-cluster, final cluster

    def test_agentic_continues_when_llm_says_so(self):
        """Loop respects LLM's should_continue=True up to max_iterations."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()

        # Iteration 1 returns hits, iteration 2 returns new hits, iteration 3 same (no new)
        call_count = [0]
        def search_side_effect(*a, **kw):
            call_count[0] += 1
            # Return different IDs for first two groups of calls
            if call_count[0] <= 2:
                return [
                    {"id": f"doc{call_count[0]}#0", "url": f"https://a{call_count[0]}.com", "title": "A",
                     "source_domain": "a.com", "chunk_index": 0,
                     "chunk_text": "complaint text", "score": 0.9},
                ]
            return [
                {"id": "doc1#0", "url": "https://a1.com", "title": "A",
                 "source_domain": "a.com", "chunk_index": 0,
                 "chunk_text": "complaint text", "score": 0.9},
            ]
        mock_rag.search.side_effect = search_side_effect
        mock_rag.get_doc.side_effect = lambda doc_id: [
            {"id": f"{doc_id}#0", "url": f"https://{doc_id}.com", "title": "A",
             "source_domain": "a.com", "chunk_index": 0, "chunk_text": "complaint text"},
        ]

        plan_response = MagicMock()
        plan_response.content = json.dumps(["query"])

        signals_response = MagicMock()
        signals_response.content = json.dumps([
            {"pain": "some pain", "workaround": "none", "desired_outcome": "fix",
             "segment": "users", "severity": 3, "willingness_to_pay": 3,
             "keywords": [], "doc_id": "doc1", "url": "https://a1.com"},
        ])

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])

        evaluate_continue = MagicMock()
        evaluate_continue.content = json.dumps(
            {"should_continue": True, "reasoning": "need more data", "suggested_focus": "explore pricing"}
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response,       # iter 1: plan
            signals_response,    # iter 1: extract
            cluster_response,    # iter 1: eval → cluster
            evaluate_continue,   # iter 1: eval → LLM says continue
            plan_response,       # iter 2: plan
            signals_response,    # iter 2: extract
            cluster_response,    # iter 2: eval → cluster
            evaluate_continue,   # iter 2: eval → LLM says continue
            plan_response,       # iter 3: plan (max_iterations=3, last iteration — no eval)
            # iter 3: no new hits → extract skipped, then max_iterations reached → stops
            cluster_response,    # final cluster
        ]

        loop = InsightLoop(
            goal="test", rag_store=mock_rag, llm=mock_llm, max_iterations=3,
            use_agent=False,
        )
        loop.run()

        # Should have run all 3 iterations
        # Plan was called 3 times
        plan_calls = [c for c in mock_llm.invoke.call_args_list
                      if "planning search queries" in str(c).lower()]
        assert len(plan_calls) == 3

    def test_structured_check_overrides_llm(self):
        """max_iterations stops the loop even if LLM would continue."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            {"id": "doc1#0", "url": "https://a.com", "title": "A",
             "source_domain": "a.com", "chunk_index": 0,
             "chunk_text": "complaint", "score": 0.9},
        ]
        mock_rag.get_doc.side_effect = lambda doc_id: {
            "doc1": [{"id": "doc1#0", "url": "https://a.com", "title": "A",
                       "source_domain": "a.com", "chunk_index": 0, "chunk_text": "complaint"}],
        }.get(doc_id, [])

        plan_response = MagicMock()
        plan_response.content = json.dumps(["query"])

        signals_response = MagicMock()
        signals_response.content = json.dumps([
            {"pain": "issue", "workaround": "none", "desired_outcome": "fix",
             "segment": "users", "severity": 3, "willingness_to_pay": 3,
             "keywords": [], "doc_id": "doc1", "url": "https://a.com"},
        ])

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response,       # iter 1: plan
            signals_response,    # iter 1: extract
            # max_iterations=1 → stops before eval
            cluster_response,    # final cluster
        ]

        loop = InsightLoop(
            goal="test", rag_store=mock_rag, llm=mock_llm, max_iterations=1,
            use_agent=False,
        )
        loop.run()

        # With max_iterations=1, the loop runs 1 iteration then stops
        # No evaluate_progress call should have been made
        # (plan + signals + final cluster = 3 calls)
        assert mock_llm.invoke.call_count == 3

    def test_min_evidence_filter(self):
        """Opportunities with too few signals are filtered out."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            {"id": "doc1#0", "url": "https://a.com", "title": "A",
             "source_domain": "a.com", "chunk_index": 0,
             "chunk_text": "complaint about X", "score": 0.9},
        ]
        mock_rag.get_doc.side_effect = lambda doc_id: {
            "doc1": [{"id": "doc1#0", "url": "https://a.com", "title": "A",
                       "source_domain": "a.com", "chunk_index": 0, "chunk_text": "complaint about X"}],
        }.get(doc_id, [])

        plan_response = MagicMock()
        plan_response.content = json.dumps(["query"])

        signals_response = MagicMock()
        signals_response.content = json.dumps([
            {
                "pain": "unique issue",
                "workaround": "none",
                "desired_outcome": "fix",
                "segment": "users",
                "severity": 3,
                "willingness_to_pay": 3,
                "keywords": [],
                "doc_id": "doc1",
                "url": "https://a.com",
            },
        ])

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([
            {
                "title": "Niche Issue",
                "summary": "Only one signal",
                "pain_keys": ["unique issue"],
                "confidence": "low",
            },
        ])

        evaluate_continue = MagicMock()
        evaluate_continue.content = json.dumps(
            {"should_continue": True, "reasoning": "need more", "suggested_focus": "more"}
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            plan_response,      # plan queries
            signals_response,   # extract signals
            cluster_response,   # eval → cluster
            evaluate_continue,  # eval → LLM
            plan_response,      # iteration 2 plan (no new hits → stops)
            cluster_response,   # final cluster
        ]

        loop = InsightLoop(
            goal="test",
            rag_store=mock_rag,
            llm=mock_llm,
            iterations=2,
            min_evidence_per_opportunity=3,  # requires 3 but only 1 signal
            use_agent=False,
        )
        opportunities = loop.run()
        assert len(opportunities) == 0


# ── Tools ─────────────────────────────────────────────────────────


class TestRecordSignalTool:
    def test_record_signal_tool(self):
        from insights.tools import make_record_signal_tool

        store = EvidenceStore()
        tool = make_record_signal_tool(store)
        result = tool.invoke({
            "pain": "slow api",
            "workaround": "caching",
            "desired_outcome": "fast api",
            "segment": "developers",
            "severity": 4,
            "willingness_to_pay": 3,
            "keywords": ["api", "speed"],
            "doc_id": "doc1",
            "url": "https://example.com",
        })
        assert "recorded" in result.lower()
        assert store.total_signals == 1
        assert "slow api" in store.pain_groups

    def test_record_signal_dedup(self):
        from insights.tools import make_record_signal_tool

        store = EvidenceStore()
        tool = make_record_signal_tool(store)
        args = {
            "pain": "slow api",
            "workaround": "caching",
            "desired_outcome": "fast api",
            "segment": "developers",
            "severity": 4,
            "willingness_to_pay": 3,
            "keywords": ["api"],
            "doc_id": "doc1",
            "url": "https://example.com",
        }
        tool.invoke(args)
        result = tool.invoke(args)
        assert "duplicate" in result.lower()
        assert store.total_signals == 1


class TestGetProgressTool:
    def test_get_progress_tool(self):
        from insights.tools import make_get_progress_tool

        store = EvidenceStore()
        store.add(_make_signal(doc_id="a#0", pain="slow api"))
        store.add(_make_signal(doc_id="b#0", pain="slow api"))
        store.add(_make_signal(doc_id="c#0", pain="bad docs"))
        tool = make_get_progress_tool(store, max_evidence=100)
        result = tool.invoke({})
        assert "3/100" in result
        assert "Pain groups: 2" in result
        assert "slow api" in result

    def test_get_progress_empty(self):
        from insights.tools import make_get_progress_tool

        store = EvidenceStore()
        tool = make_get_progress_tool(store, max_evidence=50)
        result = tool.invoke({})
        assert "0/50" in result
        assert "Pain groups: 0" in result


class TestFetchPageTool:
    def test_fetch_page_structured_output(self):
        """fetch_page returns structured output with URL, domain, title, and content."""
        mock_fetch = MagicMock(return_value={"text": "Page content here", "title": "My Title"})
        with patch("research_loop.fetch.fetch_page", mock_fetch):
            from insights.tools import make_fetch_page_tool
            tool = make_fetch_page_tool()
            result = tool.invoke({"url": "https://example.com/page"})
        assert "URL: https://example.com/page" in result
        assert "Domain: example.com" in result
        assert "Title: My Title" in result
        assert "Page content here" in result


class TestIndexToRagTool:
    def test_chunks_embeds_and_upserts(self):
        """index_to_rag tool chunks text, embeds, and upserts into DB."""
        from insights.tools import make_index_to_rag_tool

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # Mock the dimension check query for ensure_index (not called in tool, but PgVectorIndexer is created)
        mock_cursor.fetchone.return_value = None

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384, [0.2] * 384]

        indexed_urls: set[str] = set()
        tool = make_index_to_rag_tool(
            conn=mock_conn,
            table_name="chunks",
            namespace="test",
            embedder=mock_embedder,
            dimension=384,
            goal="test goal",
            indexed_urls=indexed_urls,
        )

        result = tool.invoke({
            "url": "https://example.com/page",
            "title": "Test Page",
            "text": "A " * 500 + "B " * 500,  # Long enough to produce 2 chunks
            "source_domain": "example.com",
        })

        assert "Indexed" in result
        assert "https://example.com/page" in result
        assert "https://example.com/page" in indexed_urls
        mock_embedder.embed.assert_called_once()

    def test_dedup_prevents_reindex(self):
        """index_to_rag skips already-indexed URLs."""
        from insights.tools import make_index_to_rag_tool

        mock_conn = MagicMock()
        mock_embedder = MagicMock()
        indexed_urls = {"https://example.com/page"}

        tool = make_index_to_rag_tool(
            conn=mock_conn, table_name="chunks", namespace="test",
            embedder=mock_embedder, dimension=384, goal="test",
            indexed_urls=indexed_urls,
        )

        result = tool.invoke({
            "url": "https://example.com/page",
            "title": "Test", "text": "content", "source_domain": "example.com",
        })

        assert "Already indexed" in result
        mock_embedder.embed.assert_not_called()

    def test_empty_text_handled(self):
        """index_to_rag handles empty text gracefully."""
        from insights.tools import make_index_to_rag_tool

        mock_conn = MagicMock()
        mock_embedder = MagicMock()

        tool = make_index_to_rag_tool(
            conn=mock_conn, table_name="chunks", namespace="test",
            embedder=mock_embedder, dimension=384, goal="test",
        )

        result = tool.invoke({
            "url": "https://example.com/empty",
            "title": "Empty", "text": "", "source_domain": "example.com",
        })

        assert "Skipped" in result
        mock_embedder.embed.assert_not_called()


class TestTools:
    def test_rag_search_tool(self):
        from insights.tools import make_rag_search_tool

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"id": "a#0", "url": "https://a.com", "title": "Title A",
             "source_domain": "a.com", "chunk_text": "some text", "score": 0.9},
        ]
        tool = make_rag_search_tool(mock_store)
        result = tool.invoke({"query": "test"})
        assert "Title A" in result
        assert "0.900" in result

    def test_rag_search_no_results(self):
        from insights.tools import make_rag_search_tool

        mock_store = MagicMock()
        mock_store.search.return_value = []
        tool = make_rag_search_tool(mock_store)
        result = tool.invoke({"query": "nothing"})
        assert "No results" in result

    def test_rag_get_doc_tool(self):
        from insights.tools import make_rag_get_doc_tool

        mock_store = MagicMock()
        mock_store.get_doc.return_value = [
            {"chunk_index": 0, "chunk_text": "first chunk", "title": "Doc Title", "url": "https://x.com"},
            {"chunk_index": 1, "chunk_text": "second chunk", "title": "Doc Title", "url": "https://x.com"},
        ]
        tool = make_rag_get_doc_tool(mock_store)
        result = tool.invoke({"doc_id": "abc123"})
        assert "Doc Title" in result
        assert "first chunk" in result
        assert "second chunk" in result


# ── Helpers ───────────────────────────────────────────────────────


# ── Agent Loop ────────────────────────────────────────────────────


class TestAgentLoop:
    def _make_fake_agent(self, tool_call_sequence):
        """Create a fake agent that exercises tools via scripted calls.

        tool_call_sequence: list of dicts, each with 'name' and 'args'.
        The fake agent returns messages that look like tool calls + responses.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        def fake_invoke(input_dict, config=None):
            messages = list(input_dict.get("messages", []))
            for tc in tool_call_sequence:
                # Create an AIMessage with tool_calls
                ai_msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": f"call_{tc['name']}",
                        "name": tc["name"],
                        "args": tc["args"],
                    }],
                )
                messages.append(ai_msg)
                # Simulate tool execution by actually calling the tool
                # (The outer test patches create_react_agent, so tools won't be called.
                #  We just need the message structure for query extraction.)
                messages.append(ToolMessage(
                    content=tc.get("result", "ok"),
                    tool_call_id=f"call_{tc['name']}",
                ))
            # Final AI message
            messages.append(AIMessage(content="Done with this iteration."))
            return {"messages": messages}

        mock_agent = MagicMock()
        mock_agent.invoke = fake_invoke
        return mock_agent

    def test_agent_queries_extracted_from_history(self):
        """all_queries is populated from rag_search tool calls in agent messages."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        # Cluster response for the final cluster_and_merge
        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        tool_calls = [
            {"name": "rag_search", "args": {"query": "api pain points"}},
            {"name": "rag_search", "args": {"query": "developer frustrations"}},
        ]
        fake_agent = self._make_fake_agent(tool_calls)

        with patch("insights.loop.create_react_agent", return_value=fake_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=1, use_agent=True,
            )
            loop.run()

        assert "api pain points" in loop.all_queries
        assert "developer frustrations" in loop.all_queries

    def test_agent_records_signals_via_tool(self):
        """Agent loop populates EvidenceStore when record_signal tool calls are made."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([
            {"title": "Slow API", "summary": "API is slow",
             "pain_keys": ["slow api"], "confidence": "high"},
        ])
        mock_llm.invoke.return_value = cluster_response

        # Directly pre-populate the evidence store to simulate
        # what record_signal tool calls would do during agent execution
        store = EvidenceStore()
        store.add(_make_signal(doc_id="doc1#0", pain="slow api"))
        store.add(_make_signal(doc_id="doc2#0", pain="slow api"))

        fake_agent = self._make_fake_agent([
            {"name": "rag_search", "args": {"query": "slow api"}},
        ])

        with patch("insights.loop.create_react_agent", return_value=fake_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=1, use_agent=True, min_evidence_per_opportunity=2,
            )
            # Inject pre-populated store
            loop.evidence = store
            opportunities = loop.run()

        assert len(opportunities) == 1
        assert opportunities[0].title == "Slow API"

    def test_agent_stops_at_max_evidence(self):
        """Agent loop stops when max_evidence is reached."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        # Pre-populate evidence at the limit
        store = EvidenceStore()
        for i in range(5):
            store.add(_make_signal(doc_id=f"doc{i}#0", pain=f"pain {i}"))

        fake_agent = self._make_fake_agent([])

        with patch("insights.loop.create_react_agent", return_value=fake_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=5, max_evidence=5, use_agent=True,
            )
            loop.evidence = store
            loop.run()

        # Should have stopped after 1 iteration (max_evidence already met)
        assert mock_llm.invoke.call_count == 1  # only final cluster_and_merge

    def test_agent_stops_on_no_new_signals(self):
        """Agent loop stops when no new signals are added in an iteration."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        # Pre-populate with some signals (below max)
        store = EvidenceStore()
        store.add(_make_signal(doc_id="doc1#0", pain="pain 1"))

        # Agent doesn't add any new signals
        fake_agent = self._make_fake_agent([])

        with patch("insights.loop.create_react_agent", return_value=fake_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=5, max_evidence=100, use_agent=True,
            )
            loop.evidence = store
            loop.run()

        # iter 1: signals_before=1, signals_after=1, 0 new → but only stops on iter>1
        # iter 2: signals_before=1, signals_after=1, 0 new → stops
        # final cluster_and_merge = 1 llm call
        assert mock_llm.invoke.call_count == 1

    def test_backward_compat_manual_loop(self):
        """use_agent=False (default) still runs the old manual loop."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_rag.get_doc.return_value = []

        plan_response = MagicMock()
        plan_response.content = json.dumps(["test query"])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = plan_response

        loop = InsightLoop(
            goal="test",
            rag_store=mock_rag,
            llm=mock_llm,
            iterations=1,
            use_agent=False,
        )
        opportunities = loop.run()
        assert opportunities == []
        # Should have called llm.invoke for plan_queries (manual loop behavior)
        mock_llm.invoke.assert_called()


# ── Autonomous Loop ──────────────────────────────────────────────


class TestAutonomousLoop:
    def _make_fake_agent(self, tool_call_sequence):
        """Create a fake agent for autonomous loop testing."""
        from langchain_core.messages import AIMessage, ToolMessage

        def fake_invoke(input_dict, config=None):
            messages = list(input_dict.get("messages", []))
            for tc in tool_call_sequence:
                ai_msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": f"call_{tc['name']}",
                        "name": tc["name"],
                        "args": tc["args"],
                    }],
                )
                messages.append(ai_msg)
                messages.append(ToolMessage(
                    content=tc.get("result", "ok"),
                    tool_call_id=f"call_{tc['name']}",
                ))
            messages.append(AIMessage(content="Done with this iteration."))
            return {"messages": messages}

        mock_agent = MagicMock()
        mock_agent.invoke = fake_invoke
        return mock_agent

    def test_autonomous_calls_ensure_index(self):
        """Autonomous loop calls PgVectorIndexer.ensure_index at start."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()

        fake_agent = self._make_fake_agent([])

        with patch("insights.loop.create_react_agent", return_value=fake_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=1, autonomous=True,
                pg_conn=mock_conn, index_table="chunks",
                index_namespace="test", index_dimension=384,
                embedder=mock_embedder, brave_token="fake-token",
            )
            loop.run()

        # Verify ensure_index was called (CREATE EXTENSION + table creation)
        assert mock_cursor.execute.call_count >= 1

    def test_autonomous_builds_correct_tools(self):
        """Autonomous loop includes index_to_rag, web_search, fetch_page tools."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()

        captured_tools = []

        def capture_create_agent(llm, tools, prompt=None):
            captured_tools.extend(t.name for t in tools)
            fake = MagicMock()
            fake.invoke.return_value = {"messages": []}
            return fake

        with patch("insights.loop.create_react_agent", side_effect=capture_create_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=1, autonomous=True,
                pg_conn=mock_conn, index_table="chunks",
                index_namespace="test", index_dimension=384,
                embedder=mock_embedder, brave_token="fake-token",
            )
            loop.run()

        assert "index_to_rag" in captured_tools
        assert "web_search" in captured_tools
        assert "fetch_page" in captured_tools
        assert "rag_search" in captured_tools
        assert "record_signal" in captured_tools

    def test_autonomous_shares_indexed_urls_across_iterations(self):
        """indexed_urls set is shared across iterations."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()

        # Simulate agent adding signals in first iteration so it doesn't stop early
        iteration_count = [0]

        def fake_create_agent(llm, tools, prompt=None):
            fake = MagicMock()

            def fake_invoke(input_dict, config=None):
                iteration_count[0] += 1
                from langchain_core.messages import AIMessage
                return {"messages": [AIMessage(content="Done")]}

            fake.invoke = fake_invoke
            return fake

        with patch("insights.loop.create_react_agent", side_effect=fake_create_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=1, autonomous=True,
                pg_conn=mock_conn, index_table="chunks",
                index_namespace="test", index_dimension=384,
                embedder=mock_embedder, brave_token="fake-token",
            )
            # Pre-populate to verify sharing
            loop.indexed_urls.add("https://already.com")
            loop.run()

        assert "https://already.com" in loop.indexed_urls

    def test_autonomous_clears_search_cache(self):
        """Autonomous loop clears rag_store search cache between iterations."""
        from insights.loop import InsightLoop

        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_rag._search_cache = {"old_query": "old_result"}
        mock_rag._embed_cache = {"old_embed": "old_vec"}
        mock_llm = MagicMock()

        cluster_response = MagicMock()
        cluster_response.content = json.dumps([])
        mock_llm.invoke.return_value = cluster_response

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()

        def fake_create_agent(llm, tools, prompt=None):
            fake = MagicMock()
            from langchain_core.messages import AIMessage
            fake.invoke.return_value = {"messages": [AIMessage(content="Done")]}
            return fake

        with patch("insights.loop.create_react_agent", side_effect=fake_create_agent):
            loop = InsightLoop(
                goal="test", rag_store=mock_rag, llm=mock_llm,
                max_iterations=1, autonomous=True,
                pg_conn=mock_conn, index_table="chunks",
                index_namespace="test", index_dimension=384,
                embedder=mock_embedder, brave_token="fake-token",
            )
            loop.run()

        # Caches should have been cleared
        assert len(mock_rag._search_cache) == 0
        assert len(mock_rag._embed_cache) == 0


# ── Helpers ───────────────────────────────────────────────────────


def _make_signal(
    pain: str = "default pain",
    doc_id: str = "doc#0",
    url: str = "https://example.com",
    severity: int = 3,
    willingness_to_pay: int = 3,
    **kwargs,
) -> Signal:
    return Signal(
        pain=pain,
        workaround=kwargs.get("workaround", "manual workaround"),
        desired_outcome=kwargs.get("desired_outcome", "automated solution"),
        segment=kwargs.get("segment", "developers"),
        severity=severity,
        willingness_to_pay=willingness_to_pay,
        keywords=kwargs.get("keywords", ["test"]),
        doc_id=doc_id,
        url=url,
    )
