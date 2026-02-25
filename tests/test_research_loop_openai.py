"""Tests for the research_loop_openai package."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_loop_openai.agent import (
    IterationOutput,
    ResearchAgent,
    SearchResult,
)
from research_loop_openai.loop import OpenAIResearchLoop


# ── Agent ────────────────────────────────────────────────────────────

def _make_iteration_output(
    results: list[dict] | None = None,
    summary: str = "Found relevant discussions.",
    suggested: list[str] | None = None,
) -> IterationOutput:
    if results is None:
        results = [
            {"url": "https://example.com/1", "title": "Result 1", "snippet": "snippet 1", "source_domain": "example.com"},
            {"url": "https://example.com/2", "title": "Result 2", "snippet": "snippet 2", "source_domain": "example.com"},
        ]
    return IterationOutput(
        results=[SearchResult(**r) for r in results],
        summary=summary,
        suggested_next_queries=suggested or ["follow-up query 1"],
    )


class TestResearchAgent:
    def test_run_iteration_sync(self):
        agent = ResearchAgent(goal="test goal", domains=["example.com"])

        mock_result = MagicMock()
        mock_result.final_output = _make_iteration_output()

        with patch("research_loop_openai.agent.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=mock_result)
            output = agent.run_iteration_sync(
                iteration=1,
                num_queries=3,
                previous_queries=[],
                research_summary=None,
                known_urls=[],
            )

        assert len(output.results) == 2
        assert output.results[0].url == "https://example.com/1"
        assert output.summary == "Found relevant discussions."
        assert len(output.suggested_next_queries) == 1

    def test_agent_with_custom_model(self):
        agent = ResearchAgent(goal="test", domains=["d.com"], model="gpt-4o-mini")
        assert agent.agent.model == "gpt-4o-mini"

    def test_agent_default_model_is_gpt5_mini(self):
        agent = ResearchAgent(goal="test", domains=["d.com"])
        assert agent.agent.model == "gpt-5-mini"

    def test_system_prompt_includes_goal_and_domains(self):
        agent = ResearchAgent(goal="find pain points", domains=["reddit.com", "hn.com"])
        assert "find pain points" in agent.agent.instructions
        assert "reddit.com" in agent.agent.instructions
        assert "hn.com" in agent.agent.instructions


# ── Loop Integration ─────────────────────────────────────────────────

def _patch_agent(loop: OpenAIResearchLoop, outputs: list[IterationOutput]) -> None:
    """Replace the agent's run_iteration_sync with a mock returning predefined outputs."""
    call_idx = {"n": 0}

    def fake_run_iteration_sync(**kwargs):
        idx = min(call_idx["n"], len(outputs) - 1)
        call_idx["n"] += 1
        return outputs[idx]

    loop.agent.run_iteration_sync = fake_run_iteration_sync


class TestOpenAIResearchLoop:
    def test_basic_run(self, tmp_path):
        out = tmp_path / "test.jsonl"
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=1,
            per_iter_queries=3,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, [_make_iteration_output()])
        loop.run()

        lines = out.read_text().strip().split("\n")
        records = [json.loads(l) for l in lines]
        assert len(records) == 2
        assert all(r["dedupe"] == "unique" for r in records)
        assert records[0]["url"] == "https://example.com/1"
        assert records[0]["goal"] == "test"
        assert records[0]["run_id"] == loop.run_id

    def test_max_docs_stopping(self, tmp_path):
        out = tmp_path / "max.jsonl"
        results = [
            {"url": f"https://example.com/{i}", "title": f"R{i}", "snippet": f"s{i}", "source_domain": "example.com"}
            for i in range(10)
        ]
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=5,
            per_iter_queries=3,
            max_docs=3,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, [_make_iteration_output(results=results)])
        loop.run()

        lines = out.read_text().strip().split("\n")
        unique = [json.loads(l) for l in lines if json.loads(l)["dedupe"] == "unique"]
        assert len(unique) == 3

    def test_dedup_skipping(self, tmp_path):
        out = tmp_path / "dedup.jsonl"
        # Same URL twice in results
        results = [
            {"url": "https://example.com/same", "title": "Same", "snippet": "s", "source_domain": "example.com"},
            {"url": "https://example.com/same", "title": "Same", "snippet": "s", "source_domain": "example.com"},
        ]
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=1,
            per_iter_queries=1,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, [_make_iteration_output(results=results)])
        loop.run()

        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        unique = [l for l in lines if l["dedupe"] == "unique"]
        dupes = [l for l in lines if l["dedupe"] == "duplicate"]
        assert len(unique) == 1
        assert len(dupes) == 1

    def test_dedup_across_iterations(self, tmp_path):
        out = tmp_path / "dedup_iter.jsonl"
        # Iteration 1 and 2 return the same URL
        output1 = _make_iteration_output(results=[
            {"url": "https://example.com/page", "title": "Page", "snippet": "s", "source_domain": "example.com"},
        ])
        output2 = _make_iteration_output(results=[
            {"url": "https://example.com/page", "title": "Page", "snippet": "s", "source_domain": "example.com"},
            {"url": "https://example.com/new", "title": "New", "snippet": "s2", "source_domain": "example.com"},
        ])
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=2,
            per_iter_queries=1,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, [output1, output2])
        loop.run()

        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        unique = [l for l in lines if l["dedupe"] == "unique"]
        dupes = [l for l in lines if l["dedupe"] == "duplicate"]
        assert len(unique) == 2  # /page from iter1, /new from iter2
        assert len(dupes) == 1  # /page from iter2

    def test_dry_run_skips_fetch(self, tmp_path):
        out = tmp_path / "dry.jsonl"
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=1,
            per_iter_queries=1,
            max_docs=10,
            out=str(out),
            fetch_pages=True,
            dry_run=True,
            sleep_sec=0,
        )
        _patch_agent(loop, [_make_iteration_output()])

        with patch("research_loop_openai.loop.fetch_page") as mock_fetch:
            loop.run()
            mock_fetch.assert_not_called()

        assert out.exists()

    def test_summaries_accumulate(self, tmp_path):
        out = tmp_path / "summaries.jsonl"
        outputs = [
            _make_iteration_output(
                results=[{"url": f"https://example.com/iter{i}", "title": f"R{i}", "snippet": f"s{i}", "source_domain": "example.com"}],
                summary=f"Summary iteration {i}",
            )
            for i in range(1, 4)
        ]
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=3,
            per_iter_queries=1,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, outputs)
        loop.run()

        assert len(loop.iteration_summaries) == 3
        assert "Summary iteration 1" in loop.iteration_summaries[0]

    def test_no_new_docs_stops_early(self, tmp_path):
        out = tmp_path / "early_stop.jsonl"
        # First iteration: one result. Second: same URL (dedup only, 0 new).
        output1 = _make_iteration_output(results=[
            {"url": "https://example.com/only", "title": "Only", "snippet": "s", "source_domain": "example.com"},
        ])
        output2 = _make_iteration_output(results=[
            {"url": "https://example.com/only", "title": "Only", "snippet": "s", "source_domain": "example.com"},
        ])
        output3 = _make_iteration_output(results=[
            {"url": "https://example.com/never", "title": "Never", "snippet": "s", "source_domain": "example.com"},
        ])
        loop = OpenAIResearchLoop(
            goal="test",
            domains=["example.com"],
            iterations=3,
            per_iter_queries=1,
            max_docs=100,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, [output1, output2, output3])
        loop.run()

        # Should stop after iteration 2 (0 new docs), never reaching iteration 3
        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        urls = [l["url"] for l in lines]
        assert "https://example.com/never" not in urls

    def test_jsonl_record_schema(self, tmp_path):
        out = tmp_path / "schema.jsonl"
        loop = OpenAIResearchLoop(
            goal="find bugs",
            domains=["github.com"],
            iterations=1,
            per_iter_queries=1,
            max_docs=1,
            out=str(out),
            sleep_sec=0,
        )
        _patch_agent(loop, [_make_iteration_output(results=[
            {"url": "https://github.com/issue/1", "title": "Bug", "snippet": "crash on start", "source_domain": "github.com"},
        ])])
        loop.run()

        record = json.loads(out.read_text().strip())
        required_keys = {"run_id", "ts", "goal", "domains", "iteration", "query", "rank", "url", "title", "snippet", "source_domain", "content", "dedupe", "errors"}
        assert required_keys == set(record.keys())
        assert record["goal"] == "find bugs"
        assert record["domains"] == ["github.com"]
        assert record["dedupe"] == "unique"
        assert record["errors"] == []


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_parse_args_required(self):
        from research_loop_openai.cli import parse_args

        args = parse_args(["--goal", "test goal", "--domains", "example.com"])
        assert args.goal == "test goal"
        assert args.domains == ["example.com"]
        assert args.iterations == 5
        assert args.openai_model is None

    def test_parse_args_all_options(self):
        from research_loop_openai.cli import parse_args

        args = parse_args([
            "--goal", "g",
            "--domains", "a.com", "b.com",
            "--iterations", "3",
            "--per-iter-queries", "2",
            "--results-per-query", "10",
            "--max-docs", "50",
            "--out", "/tmp/out.jsonl",
            "--fetch-pages",
            "--timeout-sec", "30",
            "--sleep-sec", "1.0",
            "--dry-run",
            "--openai-model", "gpt-4o",
            "--verbose",
        ])
        assert args.goal == "g"
        assert args.domains == ["a.com", "b.com"]
        assert args.iterations == 3
        assert args.per_iter_queries == 2
        assert args.results_per_query == 10
        assert args.max_docs == 50
        assert args.out == "/tmp/out.jsonl"
        assert args.fetch_pages is True
        assert args.timeout_sec == 30
        assert args.sleep_sec == 1.0
        assert args.dry_run is True
        assert args.openai_model == "gpt-4o"
        assert args.verbose is True
