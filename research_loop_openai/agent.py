"""OpenAI Agents SDK research agent with structured output."""

from __future__ import annotations

import asyncio
import logging

from agents import Agent, Runner, WebSearchTool
from pydantic import BaseModel

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    source_domain: str


class IterationOutput(BaseModel):
    results: list[SearchResult]
    summary: str
    suggested_next_queries: list[str]


SYSTEM_PROMPT = """\
You are a research assistant that finds diverse, high-quality web sources for a given research goal.

Research Goal: {goal}
Target domains: {domains}

Strategy:
- Search with diverse intents: pain points, feature requests, alternatives, comparisons, tutorials, reviews, discussions
- Focus on the target domains but also find relevant results from other sources
- Return structured results with accurate URLs, titles, and snippets
- Suggest follow-up queries that explore different angles from what has already been covered
- Avoid repeating previous queries or returning already-known URLs
"""

ITERATION_PROMPT = """\
Iteration {iteration} of research.

Generate {num_queries} diverse search queries and execute them to find relevant results.

Previous queries (avoid repeating): {previous_queries}

Research summary so far: {research_summary}

Known URLs (avoid returning these): {known_urls}

Search the web using your tools, then return structured results. Include the most relevant \
and unique findings. For each result provide the exact URL, page title, a useful snippet, \
and the source domain. Also suggest {num_queries} follow-up queries for the next iteration.
"""


class ResearchAgent:
    """Wraps an OpenAI Agent with WebSearchTool for iterative research."""

    def __init__(
        self,
        goal: str,
        domains: list[str],
        model: str | None = None,
    ):
        self.goal = goal
        self.domains = domains
        instructions = SYSTEM_PROMPT.format(
            goal=goal,
            domains=", ".join(domains),
        )
        agent_kwargs: dict = {
            "name": "ResearchAgent",
            "instructions": instructions,
            "tools": [WebSearchTool()],
            "output_type": IterationOutput,
            "model": model or DEFAULT_MODEL,
        }
        self.agent = Agent(**agent_kwargs)

    async def run_iteration(
        self,
        iteration: int,
        num_queries: int,
        previous_queries: list[str],
        research_summary: str | None,
        known_urls: list[str],
    ) -> IterationOutput:
        prompt = ITERATION_PROMPT.format(
            iteration=iteration,
            num_queries=num_queries,
            previous_queries=", ".join(previous_queries) if previous_queries else "none",
            research_summary=research_summary or "none yet",
            known_urls=", ".join(known_urls[-50:]) if known_urls else "none",
        )
        log.debug("Iteration %d prompt:\n%s", iteration, prompt)
        result = await Runner.run(self.agent, prompt)

        # Log run details
        turns = len(result.raw_responses)
        total_input = sum(r.usage.input_tokens for r in result.raw_responses if r.usage)
        total_output = sum(r.usage.output_tokens for r in result.raw_responses if r.usage)

        from token_tracker import get_tracker
        get_tracker().record("openai", self.agent.model, total_input, total_output)

        log.info(
            "Iteration %d agent finished: %d turn(s), %d items, tokens=%d in / %d out",
            iteration, turns, len(result.new_items), total_input, total_output,
        )

        output = result.final_output
        log.info(
            "Iteration %d results: %d URLs, %d suggested next queries",
            iteration, len(output.results), len(output.suggested_next_queries),
        )
        for i, r in enumerate(output.results):
            log.debug("  result[%d]: %s — %s", i, r.source_domain, r.url)
        if output.suggested_next_queries:
            log.debug("  next queries: %s", output.suggested_next_queries)
        log.debug("  summary: %s", output.summary)

        return output

    def run_iteration_sync(
        self,
        iteration: int,
        num_queries: int,
        previous_queries: list[str],
        research_summary: str | None,
        known_urls: list[str],
    ) -> IterationOutput:
        return asyncio.run(
            self.run_iteration(
                iteration=iteration,
                num_queries=num_queries,
                previous_queries=previous_queries,
                research_summary=research_summary,
                known_urls=known_urls,
            )
        )
