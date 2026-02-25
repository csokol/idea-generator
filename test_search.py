"""Quick CLI to test the Brave Search API directly."""

from __future__ import annotations

import argparse
import json
import os

import httpx
from dotenv import load_dotenv

from research_loop.search import BraveSearchClient

WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def web_search(query: str, count: int = 20) -> dict:
    token = os.environ.get("BRAVE_SEARCH_TOKEN", "")
    if not token:
        raise RuntimeError("BRAVE_SEARCH_TOKEN environment variable is not set.")
    headers = {"Accept": "application/json", "X-Subscription-Token": token}
    params = {"q": query, "count": min(max(count, 1), 50)}
    resp = httpx.get(WEB_SEARCH_URL, headers=headers, params=params, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"Brave API error: HTTP {resp.status_code} — {resp.text[:500]}")
    return resp.json()


def print_web_results(data: dict) -> None:
    results = data.get("web", {}).get("results", [])
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        description = r.get("description", "").strip()
        print(f"--- Result {i}: {title} ---")
        if url:
            print(f"URL: {url}")
        if description:
            print(description)
        print()


def print_llm_results(data: dict) -> None:
    results = BraveSearchClient.extract_results(data)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        text = r.get("text", "").strip()
        print(f"--- Result {i}: {title} ---")
        if url:
            print(f"URL: {url}")
        if text:
            print(text)
        print()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test Brave Search API")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", type=int, default=20, help="Number of results (default: 20)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    parser.add_argument(
        "--web", action="store_true",
        help="Use the standard web search API instead of the LLM context API",
    )
    args = parser.parse_args()

    if args.web:
        data = web_search(args.query, count=args.count)
    else:
        client = BraveSearchClient()
        data = client.search(args.query, count=args.count)

    if args.as_json:
        print(json.dumps(data, indent=2))
        return

    if args.web:
        print_web_results(data)
    else:
        print_llm_results(data)


if __name__ == "__main__":
    main()
