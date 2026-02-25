"""LLM-based query generation."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)


def _content_to_str(content) -> str:
    """Normalise LLM response content to a plain string.

    Some providers (e.g. Gemini) return a list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return str(content)

QUERY_PROMPT = """\
You are a research assistant. Generate {num_queries} diverse search queries to help \
investigate the following research goal:

Goal: {goal}
Target domains: {domains}

Iteration: {iteration}
Previous queries (avoid repeating): {previous}

Summary of findings so far: {research_summary}

Requirements:
- Each query MUST include a site: constraint for one of the target domains
- Cover diverse intents: pain points, feature requests, alternatives, comparisons, tutorials, reviews
- Be specific and varied
- Use the research summary to drill deeper into promising leads and avoid unproductive directions

Return ONLY a JSON array of objects, each with "query", "why", and "domain" fields.
Example: [{{"query": "site:example.com best plugins 2024", "why": "find popular plugins", "domain": "example.com"}}]
"""

SUMMARIZE_PROMPT = """\
You are a research assistant. Summarize the findings from a research iteration.

Research goal: {goal}

Results from this iteration:
{results}

Provide a concise summary (3-5 sentences) covering:
- Key themes and patterns found
- Gaps or areas not yet explored
- Promising leads worth investigating further

Return ONLY the summary text, no JSON or formatting.
"""


class QueryGenerator:
    def __init__(self, llm: BaseChatModel, goal: str, domains: list[str]):
        self.llm = llm
        self.goal = goal
        self.domains = domains

    def generate(
        self,
        num_queries: int = 5,
        iteration: int = 1,
        previous_queries: list[str] | None = None,
        research_summary: str | None = None,
    ) -> list[dict]:
        previous = previous_queries or []
        prompt = QUERY_PROMPT.format(
            num_queries=num_queries,
            goal=self.goal,
            domains=", ".join(self.domains),
            iteration=iteration,
            previous=json.dumps(previous) if previous else "none",
            research_summary=research_summary or "none",
        )

        log.debug("LLM request body: %s", prompt)
        response = self.llm.invoke(prompt)
        text = _content_to_str(response.content) if hasattr(response, "content") else str(response)
        log.debug("LLM response body: %s", text)

        try:
            # Try to extract JSON array from the response
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                if isinstance(queries, list) and all(isinstance(q, dict) for q in queries):
                    return queries
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: generate simple queries from goal keywords + domains
        return _fallback_queries(self.goal, self.domains, num_queries)

    def summarize_findings(self, results: list[dict]) -> str:
        if not results:
            return ""
        formatted = "\n".join(
            f"- {r.get('title', 'untitled')}: {r.get('snippet', '')[:200]}"
            for r in results
        )
        prompt = SUMMARIZE_PROMPT.format(goal=self.goal, results=formatted)
        log.debug("Summarize request: %s", prompt)
        response = self.llm.invoke(prompt)
        text = _content_to_str(response.content) if hasattr(response, "content") else str(response)
        log.debug("Summarize response: %s", text)
        return text.strip()


# Backward-compatible shims
def generate_queries(
    llm: BaseChatModel,
    goal: str,
    domains: list[str],
    num_queries: int = 5,
    iteration: int = 1,
    previous_queries: list[str] | None = None,
    research_summary: str | None = None,
) -> list[dict]:
    return QueryGenerator(llm, goal, domains).generate(
        num_queries=num_queries, iteration=iteration,
        previous_queries=previous_queries, research_summary=research_summary,
    )


def summarize_findings(
    llm: BaseChatModel,
    goal: str,
    domains: list[str],
    results: list[dict],
) -> str:
    return QueryGenerator(llm, goal, domains).summarize_findings(results)


def _fallback_queries(goal: str, domains: list[str], num_queries: int) -> list[dict]:
    words = goal.split()
    queries = []
    intents = ["problems", "best practices", "alternatives", "reviews", "how to"]
    for i in range(num_queries):
        domain = domains[i % len(domains)]
        intent = intents[i % len(intents)]
        q = f"site:{domain} {' '.join(words[:6])} {intent}"
        queries.append({"query": q, "why": f"fallback: {intent}", "domain": domain})
    return queries
