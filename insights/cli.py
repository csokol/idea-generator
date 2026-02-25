"""CLI entrypoint for the insight agent."""

from __future__ import annotations

import argparse
import logging
import sys

from index_rag.cli import DEFAULT_PG_URL
from index_rag.embedder import DEFAULT_MODEL as DEFAULT_EMBED_MODEL
from index_rag.embedder import GOOGLE_DEFAULT_DIMENSION, OPENAI_DEFAULT_DIMENSION, build_embedder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover opportunities from a RAG-indexed research corpus")
    p.add_argument("--goal", required=True, help="Research goal / topic to explore")
    p.add_argument("--namespace", default="", help="Namespace to search in")
    p.add_argument("--domains", nargs="*", help="Filter to specific source domains")
    p.add_argument("--top-k", type=int, default=10, help="Results per RAG search query")
    p.add_argument("--max-iterations", type=int, default=3, dest="max_iterations",
                    help="Maximum search-extract iterations (safety cap)")
    p.add_argument("--iterations", type=int, default=None, dest="iterations_compat",
                    help="Alias for --max-iterations (backward compat)")
    p.add_argument("--target-opportunities", type=int, default=None,
                    help="Stop when this many qualified opportunities are found")
    p.add_argument("--goal-description", default=None,
                    help="Richer goal description for the LLM agent")
    p.add_argument("--queries-per-iteration", type=int, default=5, help="Queries per iteration")
    p.add_argument("--max-evidence", type=int, default=200, help="Stop after this many signals")
    p.add_argument("--min-evidence-per-opportunity", type=int, default=2, help="Min signals to include opportunity")
    p.add_argument("--out", default=None, help="Output file path (default: stdout)")
    p.add_argument("--format", choices=["md"], default="md", help="Output format")
    p.add_argument("--pg-url", default=DEFAULT_PG_URL, help="PostgreSQL connection URL")
    p.add_argument("--index", default="rag_chunks", help="Table name")
    p.add_argument("--embed-model", default=None, help="Embedding model name")
    p.add_argument("--embed-provider", choices=["local", "google", "openai"], default="openai", help="Embedding provider")
    p.add_argument("--provider", default="gemini", help="LLM provider (groq, gemini, or openai)")
    p.add_argument("--model", default=None, help="LLM model name")
    p.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    p.add_argument("--agent", action=argparse.BooleanOptionalAction, default=True,
                    help="Use autonomous LangGraph ReAct agent (default: on; use --no-agent for manual loop)")
    p.add_argument("--autonomous", action=argparse.BooleanOptionalAction, default=False,
                    help="Autonomous mode: research + index + insights in one loop (no prior corpus needed)")
    p.add_argument("--dry-run", action="store_true", help="Connect to DB and verify, but don't run LLM")
    p.add_argument("--verbose", "-v", action="store_true", help="Structured progress logging")
    p.add_argument("--debug", "-vv", action="store_true", help="Debug logging (includes raw LLM prompts/responses)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Normalize comma-separated domains
    if args.domains:
        from research_loop.cli import parse_domains
        args.domains = parse_domains(args.domains)

    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")
    if args.debug:
        logging.getLogger("insights").setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger("insights").setLevel(logging.INFO)

    import os

    from dotenv import load_dotenv

    load_dotenv()

    brave_token = os.environ.get("BRAVE_SEARCH_TOKEN")

    if args.autonomous and not brave_token:
        print("Error: --autonomous mode requires BRAVE_SEARCH_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    import psycopg
    from pgvector.psycopg import register_vector

    from insights.loop import InsightLoop
    from insights.rag_store import RagStore
    from insights.report import generate_report
    from llm_factory import build_llm

    # Connect to Postgres
    conn = psycopg.connect(args.pg_url)
    register_vector(conn)
    print(f"Connected to PostgreSQL at {args.pg_url}")

    # Create embedder and RAG store
    dimension_map = {"google": GOOGLE_DEFAULT_DIMENSION, "openai": OPENAI_DEFAULT_DIMENSION}
    dimension = dimension_map.get(args.embed_provider)
    embedder = build_embedder(provider=args.embed_provider, model=args.embed_model, dimension=dimension)
    rag_store = RagStore(conn, args.index, args.namespace, embedder)
    print(f"RAG store: table={args.index}, namespace={args.namespace!r}")

    if args.dry_run:
        # Verify connection works with a simple query
        test_hits = rag_store.search("test", top_k=1)
        print(f"Dry-run: connection OK, test search returned {len(test_hits)} hits")
        conn.close()
        return

    # Build LLM
    llm = build_llm(provider=args.provider, model=args.model)

    # Resolve max_iterations (--iterations compat alias takes precedence if set)
    max_iters = args.iterations_compat if args.iterations_compat is not None else args.max_iterations

    # LLM log file for verbose mode
    llm_log_file = None
    if args.verbose or args.debug:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        llm_log_file = f"llm_log_{ts}.txt"
        print(f"LLM call log: {llm_log_file}")

    # Build autonomous-specific kwargs
    autonomous_kwargs = {}
    if args.autonomous:
        autonomous_kwargs = dict(
            autonomous=True,
            pg_conn=conn,
            index_table=args.index,
            index_namespace=args.namespace,
            index_dimension=dimension or OPENAI_DEFAULT_DIMENSION,
            embedder=embedder,
        )

    # Run insight loop
    loop = InsightLoop(
        goal=args.goal,
        rag_store=rag_store,
        llm=llm,
        max_iterations=max_iters,
        top_k=args.top_k,
        queries_per_iteration=args.queries_per_iteration,
        max_evidence=args.max_evidence,
        min_evidence_per_opportunity=args.min_evidence_per_opportunity,
        target_opportunities=args.target_opportunities,
        goal_description=args.goal_description,
        use_agent=args.agent,
        brave_token=brave_token,
        llm_log_file=llm_log_file,
        **autonomous_kwargs,
    )
    opportunities = loop.run()

    # Generate report
    from token_tracker import get_tracker as _get_tracker
    report = generate_report(
        opportunities,
        goal=args.goal,
        queries_explored=loop.all_queries,
        format=args.format,
        token_usage=_get_tracker().to_dict(),
    )

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.out}")
    else:
        print("\n" + report)

    conn.close()

    from token_tracker import get_tracker

    tracker = get_tracker()
    if tracker.has_usage():
        print(f"\n{tracker.summary()}")


if __name__ == "__main__":
    main()
