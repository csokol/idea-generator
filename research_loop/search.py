"""Brave Search API wrapper for the research loop."""

from __future__ import annotations

import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.search.brave.com/res/v1/web/search"
TIMEOUT = 25


class BraveSearchClient:
    def __init__(self, token: str | None = None, timeout: int = TIMEOUT, api_url: str = API_URL):
        self.token = token if token is not None else os.environ.get("BRAVE_SEARCH_TOKEN", "")
        self.timeout = timeout
        self.api_url = api_url

    def search(self, query: str, count: int = 20, freshness: str | None = None) -> dict:
        if not self.token:
            raise RuntimeError("BRAVE_SEARCH_TOKEN environment variable is not set.")

        headers = {"Accept": "application/json", "X-Subscription-Token": self.token}
        params: dict = {"q": query, "count": min(max(count, 1), 20), "extra_snippets": True}
        if freshness:
            params["freshness"] = freshness
        log.debug("Brave API request: GET %s params=%s", self.api_url, json.dumps(params))
        resp = httpx.get(self.api_url, headers=headers, params=params, timeout=self.timeout)
        log.debug("Brave API response body: %s", resp.text)
        if resp.status_code != 200:
            raise RuntimeError(f"Brave API error: HTTP {resp.status_code} — {resp.text[:500]}")
        return resp.json()

    @staticmethod
    def extract_results(data: dict) -> list[dict]:
        return data.get("web", {}).get("results", [])


# Backward-compatible shims
def brave_search(query: str, count: int = 20, timeout: int = TIMEOUT, freshness: str | None = None) -> dict:
    return BraveSearchClient(timeout=timeout).search(query, count=count, freshness=freshness)


def extract_results(data: dict) -> list[dict]:
    return BraveSearchClient.extract_results(data)
