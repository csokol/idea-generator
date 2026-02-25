"""Core research loop orchestration."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from llm_factory import build_llm
from research_loop.fetch import fetch_page
from research_loop.query_gen import QueryGenerator, generate_queries
from research_loop.records import Deduplicator, append_jsonl, build_record
from research_loop.search import brave_search, extract_results
from slug import generate_slug, model_tag


def _make_run_id(slug: str, mtag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}_{mtag}"


class ResearchLoop:
    def __init__(
        self,
        *,
        goal: str,
        domains: list[str],
        iterations: int = 5,
        per_iter_queries: int = 5,
        results_per_query: int = 5,
        max_docs: int = 100,
        out: str = "research.jsonl",
        fetch_pages: bool = False,
        timeout_sec: int = 10,
        sleep_sec: float = 0.5,
        user_agent: str = "ResearchLoop/0.1 (compatible; bot)",
        dry_run: bool = False,
        freshness: str | None = None,
        model: str | None = None,
        provider: str = "gemini",
    ):
        self.goal = goal
        self.domains = domains
        self.iterations = iterations
        self.per_iter_queries = per_iter_queries
        self.results_per_query = results_per_query
        self.max_docs = max_docs
        self.out = out
        self.fetch_pages = fetch_pages
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec
        self.user_agent = user_agent
        self.dry_run = dry_run
        self.freshness = freshness

        self.llm = build_llm(provider=provider, model=model)
        self.run_id = _make_run_id(generate_slug(goal, self.llm), model_tag(provider, model))
        self.dedup = Deduplicator()
        self.total_docs = 0
        self.all_queries: list[str] = []
        self.iteration_summaries: list[str] = []
        self.query_gen = QueryGenerator(self.llm, self.goal, self.domains)

    def run(self) -> None:
        self._print_header()
        for iteration in range(1, self.iterations + 1):
            if self.total_docs >= self.max_docs:
                print(f"Reached max_docs ({self.max_docs}), stopping.")
                break
            self._run_iteration(iteration)
        self._print_summary()

    def _print_header(self) -> None:
        print(f"Research loop started — run_id={self.run_id}")
        print(f"Goal: {self.goal}")
        print(f"Domains: {', '.join(self.domains)}")
        print(f"Output: {self.out}")
        print()

    def _print_summary(self) -> None:
        print(f"Done. Total documents: {self.total_docs}")
        print(f"Output written to: {self.out}")

    def _run_iteration(self, iteration: int) -> None:
        print(f"=== Iteration {iteration}/{self.iterations} ===")

        research_summary = "\n\n".join(self.iteration_summaries) if self.iteration_summaries else None

        queries = self.query_gen.generate(
            num_queries=self.per_iter_queries,
            iteration=iteration,
            previous_queries=self.all_queries,
            research_summary=research_summary,
        )

        iteration_results: list[dict] = []
        for qi, q_info in enumerate(queries):
            if self.total_docs >= self.max_docs:
                break
            self._process_query(iteration, qi, q_info, len(queries), iteration_results)

        if iteration_results:
            summary = self.query_gen.summarize_findings(iteration_results)
            self.iteration_summaries.append(summary)
            print(f"  Iteration summary: {summary}")

        print(f"  Documents collected so far: {self.total_docs}")
        print()

    def _process_query(self, iteration: int, qi: int, q_info: dict, total_queries: int, iteration_results: list[dict] | None = None) -> None:
        query = q_info["query"]
        self.all_queries.append(query)
        print(f"  [{qi + 1}/{total_queries}] {query}")

        try:
            data = brave_search(query, count=self.results_per_query, timeout=self.timeout_sec, freshness=self.freshness)
            results = extract_results(data)
        except RuntimeError as exc:
            print(f"    Search error: {exc}")
            return

        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)

        for rank, result in enumerate(results):
            if self.total_docs >= self.max_docs:
                break
            self._process_result(iteration, query, rank, result, iteration_results)

    def _process_result(self, iteration: int, query: str, rank: int, result: dict, iteration_results: list[dict] | None = None) -> None:
        url = result.get("url", "")
        title = result.get("title", "")
        description = result.get("description", "")
        extra_snippets = result.get("extra_snippets", [])
        snippet = " ".join(filter(None, [description] + extra_snippets))
        source_domain = urlparse(url).hostname or ""

        if self.dedup.is_duplicate(url):
            record = build_record(
                run_id=self.run_id,
                goal=self.goal,
                domains=self.domains,
                iteration=iteration,
                query=query,
                rank=rank,
                url=url,
                title=title,
                snippet=snippet,
                source_domain=source_domain,
                dedupe="duplicate",
            )
            append_jsonl(record, self.out)
            return

        self.dedup.mark_seen(url)
        content = None
        errors: list[str] = []

        if self.fetch_pages and not self.dry_run:
            page = fetch_page(url, timeout=self.timeout_sec, user_agent=self.user_agent)
            if page["fetched"]:
                content = page["text"]
            else:
                errors.append(page.get("error", "fetch failed"))

            if self.sleep_sec > 0:
                time.sleep(self.sleep_sec)

        record = build_record(
            run_id=self.run_id,
            goal=self.goal,
            domains=self.domains,
            iteration=iteration,
            query=query,
            rank=rank,
            url=url,
            title=title,
            snippet=snippet,
            source_domain=source_domain,
            content=content,
            errors=errors,
        )
        append_jsonl(record, self.out)
        self.total_docs += 1
        if iteration_results is not None:
            iteration_results.append({"title": title, "snippet": snippet, "url": url})


# Backward-compatible shim
def run_loop(**kwargs) -> None:
    ResearchLoop(**kwargs).run()
