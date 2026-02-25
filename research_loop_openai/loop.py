"""OpenAI Agents SDK research loop orchestrator."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

log = logging.getLogger(__name__)

from research_loop.fetch import fetch_page
from research_loop.records import Deduplicator, append_jsonl, build_record
from slug import generate_slug_openai

from .agent import DEFAULT_MODEL, ResearchAgent, SearchResult


def _make_run_id(slug: str, mtag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}_{mtag}"


class OpenAIResearchLoop:
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
        model: str | None = None,
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

        model_name = model or DEFAULT_MODEL
        self.run_id = _make_run_id(generate_slug_openai(goal), f"openai-{model_name}")
        self.agent = ResearchAgent(goal, domains, model=model)
        self.dedup = Deduplicator()
        self.total_docs = 0
        self.all_queries: list[str] = []
        self.known_urls: list[str] = []
        self.iteration_summaries: list[str] = []

    def run(self) -> None:
        self._print_header()
        for iteration in range(1, self.iterations + 1):
            if self.total_docs >= self.max_docs:
                print(f"Reached max_docs ({self.max_docs}), stopping.")
                break
            new_docs = self._run_iteration(iteration)
            if new_docs == 0 and iteration > 1:
                print("No new documents found, stopping early.")
                break
        self._print_summary()

    def _print_header(self) -> None:
        print(f"Research loop (OpenAI) started — run_id={self.run_id}")
        print(f"Goal: {self.goal}")
        print(f"Domains: {', '.join(self.domains)}")
        print(f"Output: {self.out}")
        print()

    def _print_summary(self) -> None:
        print(f"Done. Total documents: {self.total_docs}")
        print(f"Output written to: {self.out}")

    def _run_iteration(self, iteration: int) -> int:
        print(f"=== Iteration {iteration}/{self.iterations} ===")

        research_summary = (
            "\n\n".join(self.iteration_summaries) if self.iteration_summaries else None
        )

        log.info("Starting agent iteration %d (known_urls=%d, previous_queries=%d)",
                 iteration, len(self.known_urls), len(self.all_queries))

        output = self.agent.run_iteration_sync(
            iteration=iteration,
            num_queries=self.per_iter_queries,
            previous_queries=self.all_queries,
            research_summary=research_summary,
            known_urls=self.known_urls,
        )

        # Track suggested queries for next iteration
        self.all_queries.extend(output.suggested_next_queries)

        docs_before = self.total_docs
        for rank, result in enumerate(output.results):
            if self.total_docs >= self.max_docs:
                break
            self._process_result(iteration, result, rank)

        if output.summary:
            self.iteration_summaries.append(output.summary)
            print(f"  Iteration summary: {output.summary}")

        new_docs = self.total_docs - docs_before
        print(f"  Documents collected so far: {self.total_docs}")
        print()
        return new_docs

    def _process_result(
        self, iteration: int, result: SearchResult, rank: int
    ) -> None:
        url = result.url
        title = result.title
        snippet = result.snippet
        source_domain = result.source_domain or urlparse(url).hostname or ""

        # Use the agent's iteration as query context
        query = f"openai-agent-iteration-{iteration}"

        if self.dedup.is_duplicate(url):
            log.debug("Duplicate URL skipped: %s", url)
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
        self.known_urls.append(url)
        log.debug("New URL [%d]: %s — %s", rank, title, url)
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


def run_loop(**kwargs) -> None:
    OpenAIResearchLoop(**kwargs).run()
