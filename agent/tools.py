"""Brave Search wrapped as a LangChain tool for the agent."""

from __future__ import annotations

import os

import httpx
from langchain_core.tools import tool

from brave_search.cli import API_URL, TIMEOUT, format_output


def _brave_search(query: str, count: int = 10) -> str:
    """Run a Brave Search query and return human-readable results."""
    token = os.environ.get("BRAVE_SEARCH_TOKEN", "")
    if not token:
        raise RuntimeError("BRAVE_SEARCH_TOKEN environment variable is not set.")

    headers = {"Accept": "application/json", "X-Subscription-Token": token}
    params = {"q": query, "count": count}
    resp = httpx.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)

    if resp.status_code != 200:
        body = resp.text[:500]
        raise RuntimeError(f"Brave Search API error: HTTP {resp.status_code}\n{body}")

    return format_output(resp.json(), mode="human")


@tool
def brave_search(query: str) -> str:
    """Search the web using Brave Search. Use this to find current information about any topic, news, people, or events."""
    return _brave_search(query)
