"""CLI entrypoint for the OpenAI Agents SDK research loop."""

from __future__ import annotations

import argparse
import logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-loop-openai",
        description="Iterative research loop using OpenAI Agents SDK with WebSearchTool",
    )
    parser.add_argument("--goal", required=True, help="Research goal description")
    parser.add_argument("--domains", nargs="+", required=True, help="Target domains to search")
    parser.add_argument("--iterations", type=int, default=5, help="Number of research iterations (default: 5)")
    parser.add_argument("--per-iter-queries", type=int, default=5, help="Queries per iteration (default: 5)")
    parser.add_argument("--results-per-query", type=int, default=5, help="Results per query (default: 5)")
    parser.add_argument("--max-docs", type=int, default=100, help="Maximum documents to collect (default: 100)")
    parser.add_argument("--out", default="research.jsonl", help="Output JSONL file path (default: research.jsonl)")
    parser.add_argument("--fetch-pages", action="store_true", help="Fetch and extract page content")
    parser.add_argument("--timeout-sec", type=int, default=15, help="HTTP timeout in seconds (default: 15)")
    parser.add_argument("--sleep-sec", type=float, default=0.5, help="Sleep between requests (default: 0.5)")
    parser.add_argument("--user-agent", default="ResearchLoop/0.1 (compatible; bot)", help="User-Agent header")
    parser.add_argument("--dry-run", action="store_true", help="Generate queries and search, but skip page fetching")
    parser.add_argument("--openai-model", default=None, help="OpenAI model name (default: gpt-5-mini)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        # Our own modules: full DEBUG (prompts, per-result details)
        for name in ("research_loop_openai",):
            logging.getLogger(name).setLevel(logging.DEBUG)
        # OpenAI Agents SDK: INFO level (agent turns, tool calls)
        logging.getLogger("openai.agents").setLevel(logging.INFO)

    from research_loop.cli import parse_domains
    from research_loop_openai.loop import run_loop

    run_loop(
        goal=args.goal,
        domains=parse_domains(args.domains),
        iterations=args.iterations,
        per_iter_queries=args.per_iter_queries,
        results_per_query=args.results_per_query,
        max_docs=args.max_docs,
        out=args.out,
        fetch_pages=args.fetch_pages,
        timeout_sec=args.timeout_sec,
        sleep_sec=args.sleep_sec,
        user_agent=args.user_agent,
        dry_run=args.dry_run,
        model=args.openai_model,
    )

    from token_tracker import get_tracker

    tracker = get_tracker()
    if tracker.has_usage():
        print(f"\n{tracker.summary()}")


if __name__ == "__main__":
    main()
