"""Brave LLM Context API CLI tool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx

API_URL = "https://api.search.brave.com/res/v1/llm/context"
TIMEOUT = 25


def get_token() -> str:
    token = os.environ.get("BRAVE_SEARCH_TOKEN", "")
    if not token:
        print("Error: BRAVE_SEARCH_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(2)
    return token


def build_params(args: argparse.Namespace) -> dict[str, str | int]:
    count = min(max(args.count, 1), 50)

    params: dict[str, str | int] = {"q": args.query, "count": count}
    if args.country:
        params["country"] = args.country
    if args.search_lang:
        params["search_lang"] = args.search_lang
    if args.max_urls:
        params["maximum_number_of_urls"] = min(max(args.max_urls, 1), 50)
    if args.max_tokens:
        params["maximum_number_of_tokens"] = min(max(args.max_tokens, 1024), 32768)
    if args.max_snippets:
        params["maximum_number_of_snippets"] = min(max(args.max_snippets, 1), 100)
    if args.threshold_mode:
        params["context_threshold_mode"] = args.threshold_mode
    return params


def fetch_results(token: str, params: dict[str, str | int]) -> dict[str, Any]:
    headers = {"Accept": "application/json", "X-Subscription-Token": token}
    resp = httpx.get(API_URL, headers=headers, params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        body = resp.text[:500]
        print(f"Error: HTTP {resp.status_code}\n{body}", file=sys.stderr)
        sys.exit(3)
    return resp.json()


def format_output(data: dict[str, Any], *, mode: str) -> str:
    if mode == "raw":
        return json.dumps(data, indent=2, ensure_ascii=False)

    results = data.get("grounding", {}).get("generic", [])
    if not results:
        return "No results found."

    if mode == "json":
        items = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippets": r.get("snippets", [])}
            for r in results
        ]
        return json.dumps(items, ensure_ascii=False)

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        snippets = r.get("snippets", [])
        snippet_text = "\n   ".join(snippets) if snippets else "(no snippets)"
        lines.append(f"{i}. {title}\n   {url}\n   {snippet_text}")
    return "\n\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="brave-search", description="Search the web via Brave LLM Context API")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", type=int, default=20, help="Number of results (max 50)")
    parser.add_argument("--country", help="Country code filter")
    parser.add_argument("--search-lang", help="Search language")
    parser.add_argument("--max-urls", type=int, default=None, help="Maximum number of URLs (1-50)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum number of tokens (1024-32768)")
    parser.add_argument("--max-snippets", type=int, default=None, help="Maximum number of snippets (1-100)")
    parser.add_argument("--threshold-mode", choices=["strict", "balanced", "lenient", "disabled"], default=None, help="Context threshold mode")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="Output as JSON array")
    parser.add_argument("--raw", action="store_true", help="Output raw API response")
    parser.add_argument("--no-save", action="store_true", help="Do not save results or log the search")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    token = get_token()
    params = build_params(args)
    data = fetch_results(token, params)

    if args.raw:
        mode = "raw"
    elif args.json_mode:
        mode = "json"
    else:
        mode = "human"

    print(format_output(data, mode=mode))

    if not args.no_save:
        from brave_search.storage import append_log, save_result

        results = data.get("grounding", {}).get("generic", [])
        result_file = save_result(data, args.query)
        append_log(args.query, params, len(results), result_file)


if __name__ == "__main__":
    main()
