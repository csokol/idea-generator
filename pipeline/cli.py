"""CLI entrypoint for the end-to-end research → index → insights pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone

from index_rag.cli import DEFAULT_PG_URL
from index_rag.embedder import DEFAULT_DIMENSION, GOOGLE_DEFAULT_DIMENSION, OPENAI_DEFAULT_DIMENSION

STEPS = ("research", "index", "insights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Run the full research → RAG index → insights pipeline",
    )

    # --- shared / required ---
    p.add_argument("--goal", required=True, help="Research goal (used by all steps)")
    p.add_argument("--domains", nargs="+", required=True, help="Target domains")

    # --- run ID ---
    p.add_argument("--run-id", default=None, help="Pipeline run UUID (auto-generated if omitted)")

    # --- output paths ---
    p.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    p.add_argument("--out-corpus", default=None, help="JSONL corpus output path (default: data/<run-id>/corpus.jsonl)")
    p.add_argument("--out-report", default=None, help="Report output path (default: data/<run-id>/report.md)")

    # --- research loop ---
    rg = p.add_argument_group("research loop")
    rg.add_argument("--iterations", type=int, default=5, help="Research loop iterations (default: 5)")
    rg.add_argument("--per-iter-queries", type=int, default=5, help="Queries per iteration (default: 5)")
    rg.add_argument("--results-per-query", type=int, default=5, help="Results per query (default: 5)")
    rg.add_argument("--max-docs", type=int, default=100, help="Max documents to collect (default: 100)")
    rg.add_argument("--fetch-pages", action="store_true", help="Fetch and extract page content")
    rg.add_argument(
        "--freshness",
        default=None,
        help="Filter results by age: pd (24h), pw (7d), pm (31d), py (1y), or YYYY-MM-DDtoYYYY-MM-DD",
    )
    rg.add_argument("--openai-agent", action="store_true",
                    help="Use OpenAI Agents SDK research loop instead of Brave Search + LangChain")
    rg.add_argument("--openai-model", default=None,
                    help="OpenAI model name when using --openai-agent (default: gpt-5-mini)")
    rg.add_argument("--research-provider", default="gemini", help="LLM provider for research (default: gemini)")
    rg.add_argument("--research-model", default=None, help="LLM model for research")

    # --- RAG indexer ---
    ig = p.add_argument_group("RAG indexer")
    ig.add_argument("--pg-url", default=DEFAULT_PG_URL, help="PostgreSQL connection URL")
    ig.add_argument("--index", default="rag_chunks", help="Table name for vectors")
    ig.add_argument("--namespace", default=None, help="Namespace for vectors (default: run-id)")
    ig.add_argument("--embed-provider", choices=["local", "google", "openai"], default="openai", help="Embedding provider")
    ig.add_argument("--embed-model", default=None, help="Embedding model name")
    ig.add_argument("--chunk-size", type=int, default=900)
    ig.add_argument("--chunk-overlap", type=int, default=120)

    # --- insights ---
    sg = p.add_argument_group("insights")
    sg.add_argument("--insight-max-iterations", type=int, default=3, dest="insight_max_iterations",
                    help="Max insight loop iterations (default: 3)")
    sg.add_argument("--insight-iterations", type=int, default=None, dest="insight_iterations_compat",
                    help="Alias for --insight-max-iterations (backward compat)")
    sg.add_argument("--target-opportunities", type=int, default=None,
                    help="Stop insights when this many qualified opportunities are found")
    sg.add_argument("--goal-description", default=None,
                    help="Richer goal description for the insight agent")
    sg.add_argument("--insight-agent", action=argparse.BooleanOptionalAction, default=True,
                    help="Use autonomous LangGraph ReAct agent for insights (default: on; use --no-insight-agent for manual loop)")
    sg.add_argument("--autonomous", action=argparse.BooleanOptionalAction, default=False,
                    help="Autonomous insights: research + index + insights in one loop (no prior corpus needed)")
    sg.add_argument("--insight-provider", default="gemini", help="LLM provider for insights (default: gemini)")
    sg.add_argument("--insight-model", default=None, help="LLM model for insights")
    sg.add_argument("--top-k", type=int, default=10, help="Results per RAG query (default: 10)")
    sg.add_argument("--queries-per-iteration", type=int, default=5, help="Queries per insight iteration (default: 5)")
    sg.add_argument("--max-evidence", type=int, default=200, help="Stop after this many signals (default: 200)")

    # --- global ---
    p.add_argument("--start-from", choices=STEPS, default="research",
                    help="Resume from a specific step (default: research)")
    p.add_argument("--dry-run", action="store_true", help="Run research loop only, skip DB + insights")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = p.parse_args(argv)

    # Normalize comma-separated domains
    from research_loop.cli import parse_domains
    args.domains = parse_domains(args.domains)

    # run_id is finalised in main() (may need an LLM call for slug generation).
    # Fill in dependent defaults after run_id is set.
    return args


def _finalise_run_id(args: argparse.Namespace) -> None:
    """Set run_id (with LLM slug if needed) and derived paths."""
    if args.run_id is None:
        from slug import generate_slug, model_tag, _fallback_slug
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if args.openai_agent:
            from research_loop_openai.agent import DEFAULT_MODEL as OAI_DEFAULT
            mtag = f"openai-{args.openai_model or OAI_DEFAULT}"
        else:
            mtag = model_tag(args.research_provider, args.research_model)
        try:
            from llm_factory import build_llm
            llm = build_llm(provider=args.research_provider, model=args.research_model)
            slug = generate_slug(args.goal, llm)
        except Exception:
            slug = _fallback_slug(args.goal)
        args.run_id = f"{ts}_{slug}_{mtag}"

    args.run_dir = os.path.join(args.data_dir, args.run_id)
    if args.out_corpus is None:
        args.out_corpus = os.path.join(args.run_dir, "corpus.jsonl")
    if args.out_report is None:
        args.out_report = os.path.join(args.run_dir, "report.md")
    if args.namespace is None:
        args.namespace = args.run_id


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _build_resume_cmd(args: argparse.Namespace, start_from: str) -> str:
    """Build the CLI command to resume the pipeline from a given step."""
    parts = ["uv run pipeline"]
    parts.append(f"--goal {shlex.quote(args.goal)}")
    parts.append(f"--domains {' '.join(shlex.quote(d) for d in args.domains)}")
    parts.append(f"--run-id {shlex.quote(args.run_id)}")
    if args.data_dir != "data":
        parts.append(f"--data-dir {shlex.quote(args.data_dir)}")
    parts.append(f"--start-from {start_from}")

    # Research params (needed when resuming from research step)
    if start_from == "research":
        if args.openai_agent:
            parts.append("--openai-agent")
            if args.openai_model:
                parts.append(f"--openai-model {shlex.quote(args.openai_model)}")

    # RAG / shared params (needed for index and insights steps)
    if start_from in ("index", "insights"):
        if args.pg_url != DEFAULT_PG_URL:
            parts.append(f"--pg-url {shlex.quote(args.pg_url)}")
        if args.index != "rag_chunks":
            parts.append(f"--index {shlex.quote(args.index)}")
        if args.namespace != args.run_id:
            parts.append(f"--namespace {shlex.quote(args.namespace)}")
        if args.embed_provider != "openai":
            parts.append(f"--embed-provider {args.embed_provider}")
        if args.embed_model:
            parts.append(f"--embed-model {shlex.quote(args.embed_model)}")

    if start_from == "index":
        if args.chunk_size != 900:
            parts.append(f"--chunk-size {args.chunk_size}")
        if args.chunk_overlap != 120:
            parts.append(f"--chunk-overlap {args.chunk_overlap}")

    if start_from == "insights":
        effective_iters = args.insight_iterations_compat if args.insight_iterations_compat is not None else args.insight_max_iterations
        if effective_iters != 3:
            parts.append(f"--insight-max-iterations {effective_iters}")
        if args.target_opportunities is not None:
            parts.append(f"--target-opportunities {args.target_opportunities}")
        if args.goal_description:
            parts.append(f"--goal-description {shlex.quote(args.goal_description)}")
        if args.insight_provider != "gemini":
            parts.append(f"--insight-provider {shlex.quote(args.insight_provider)}")
        if args.insight_model:
            parts.append(f"--insight-model {shlex.quote(args.insight_model)}")
        if args.top_k != 10:
            parts.append(f"--top-k {args.top_k}")
        if args.queries_per_iteration != 5:
            parts.append(f"--queries-per-iteration {args.queries_per_iteration}")
        if args.max_evidence != 200:
            parts.append(f"--max-evidence {args.max_evidence}")

    if args.verbose:
        parts.append("--verbose")

    return " \\\n  ".join(parts)


def _print_failure(step: str, error: Exception, args: argparse.Namespace) -> None:
    """Print error details and the command to resume from the failed step."""
    next_step = step  # resume from the step that failed
    print(f"\n{'!' * 60}", file=sys.stderr)
    print(f"  Pipeline failed during: {step}", file=sys.stderr)
    print(f"  Error: {error}", file=sys.stderr)
    print(f"{'!' * 60}\n", file=sys.stderr)
    print("To resume from this step, run:\n", file=sys.stderr)
    print(f"  {_build_resume_cmd(args, next_step)}\n", file=sys.stderr)


def _write_params(args: argparse.Namespace, argv_raw: list[str] | None) -> None:
    """Write run parameters to data/<run-id>/params.json."""
    os.makedirs(args.run_dir, exist_ok=True)
    params = {
        "run_id": args.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(a) for a in (argv_raw or sys.argv)),
        "goal": args.goal,
        "domains": args.domains,
        "iterations": args.iterations,
        "per_iter_queries": args.per_iter_queries,
        "results_per_query": args.results_per_query,
        "max_docs": args.max_docs,
        "freshness": args.freshness,
        "openai_agent": args.openai_agent,
        "openai_model": args.openai_model,
        "research_provider": args.research_provider,
        "research_model": args.research_model,
        "index": args.index,
        "namespace": args.namespace,
        "embed_provider": args.embed_provider,
        "embed_model": args.embed_model,
        "insight_provider": args.insight_provider,
        "insight_model": args.insight_model,
        "insight_agent": getattr(args, "insight_agent", False),
        "autonomous": getattr(args, "autonomous", False),
        "insight_max_iterations": args.insight_iterations_compat if args.insight_iterations_compat is not None else args.insight_max_iterations,
        "target_opportunities": args.target_opportunities,
        "goal_description": args.goal_description,
        "start_from": args.start_from,
    }
    path = os.path.join(args.run_dir, "params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")
    print(f"  Params: {path}")


def _run_research(args: argparse.Namespace) -> int:
    if args.openai_agent:
        return _run_research_openai(args)
    return _run_research_brave(args)


def _run_research_brave(args: argparse.Namespace) -> int:
    from research_loop.loop import ResearchLoop

    loop = ResearchLoop(
        goal=args.goal,
        domains=args.domains,
        iterations=args.iterations,
        per_iter_queries=args.per_iter_queries,
        results_per_query=args.results_per_query,
        max_docs=args.max_docs,
        out=args.out_corpus,
        fetch_pages=args.fetch_pages,
        dry_run=False,
        freshness=args.freshness,
        provider=args.research_provider,
        model=args.research_model,
    )
    loop.run()
    return loop.total_docs


def _run_research_openai(args: argparse.Namespace) -> int:
    from research_loop_openai.loop import OpenAIResearchLoop

    loop = OpenAIResearchLoop(
        goal=args.goal,
        domains=args.domains,
        iterations=args.iterations,
        per_iter_queries=args.per_iter_queries,
        results_per_query=args.results_per_query,
        max_docs=args.max_docs,
        out=args.out_corpus,
        fetch_pages=args.fetch_pages,
        dry_run=False,
        model=args.openai_model,
    )
    loop.run()
    return loop.total_docs


def _run_index(args: argparse.Namespace, conn) -> int:
    from index_rag.chunker import chunk_text
    from index_rag.embedder import build_embedder
    from index_rag.ids import chunk_id, doc_id
    from index_rag.indexer import build_vector, ensure_index, upsert_batched
    from index_rag.reader import extract_text, read_jsonl

    # 1. Read and filter
    docs: list[tuple[dict, str]] = []
    skipped = 0
    for record in read_jsonl(args.out_corpus):
        text = extract_text(record, min_text_len=200)
        if text is None:
            skipped += 1
            continue
        docs.append((record, text))

    print(f"Documents: {len(docs)} usable, {skipped} skipped")

    if not docs:
        print("No usable documents to index.")
        return 0

    # 2. Chunk
    all_vectors_meta: list[tuple[str, str, dict]] = []
    for record, text in docs:
        did = doc_id(record["url"])
        chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, max_chunks_per_doc=50)
        for i, ct in enumerate(chunks):
            cid = chunk_id(did, i)
            all_vectors_meta.append((cid, ct, record))

    print(f"Chunks: {len(all_vectors_meta)} total vectors to index")

    if not all_vectors_meta:
        print("Nothing to index.")
        return 0

    # 3. Embed
    dimension = GOOGLE_DEFAULT_DIMENSION
    if args.embed_provider == "local":
        dimension = DEFAULT_DIMENSION
    elif args.embed_provider == "openai":
        dimension = OPENAI_DEFAULT_DIMENSION

    embedder = build_embedder(
        provider=args.embed_provider,
        model=args.embed_model,
        dimension=dimension if args.embed_provider in ("google", "openai") else None,
    )
    print(f"Embedding with provider={args.embed_provider!r} model={args.embed_model or 'default'}...")
    texts_to_embed = [ct for _, ct, _ in all_vectors_meta]
    embeddings = embedder.embed(texts_to_embed)

    # 4. Build vectors
    vectors = []
    for (vid, ct, record), emb in zip(all_vectors_meta, embeddings):
        vectors.append(
            build_vector(
                vid,
                emb,
                url=record.get("url", ""),
                title=record.get("title", ""),
                source_domain=record.get("source_domain", ""),
                goal=record.get("goal", ""),
                query=record.get("query", ""),
                chunk_index=int(vid.split("#")[1]),
                chunk_text=ct,
            )
        )

    # 5. Upsert
    ensure_index(conn, args.index, dimension)
    count = upsert_batched(conn, args.index, vectors, namespace=args.namespace, batch_size=100)
    print(f"Upserted {count} vectors into '{args.index}' namespace='{args.namespace}'")
    return count


def _run_insights(args: argparse.Namespace, conn) -> None:
    from index_rag.embedder import build_embedder

    from insights.loop import InsightLoop
    from insights.rag_store import RagStore
    from insights.report import generate_report
    from llm_factory import build_llm

    dimension_map = {"google": GOOGLE_DEFAULT_DIMENSION, "openai": OPENAI_DEFAULT_DIMENSION}
    dimension = dimension_map.get(args.embed_provider)
    embedder = build_embedder(provider=args.embed_provider, model=args.embed_model, dimension=dimension)
    rag_store = RagStore(conn, args.index, args.namespace, embedder)
    llm = build_llm(provider=args.insight_provider, model=args.insight_model)

    insight_max_iters = args.insight_iterations_compat if args.insight_iterations_compat is not None else args.insight_max_iterations

    autonomous_kwargs = {}
    if getattr(args, "autonomous", False):
        brave_token = os.environ.get("BRAVE_SEARCH_TOKEN")
        if not brave_token:
            raise RuntimeError("--autonomous mode requires BRAVE_SEARCH_TOKEN env var")
        autonomous_kwargs = dict(
            autonomous=True,
            pg_conn=conn,
            index_table=args.index,
            index_namespace=args.namespace,
            index_dimension=dimension or OPENAI_DEFAULT_DIMENSION,
            embedder=embedder,
            brave_token=brave_token,
        )

    # LLM log file for verbose mode
    llm_log_file = None
    if args.verbose:
        llm_log_file = os.path.join(args.run_dir, "llm_log.txt")
        print(f"  LLM call log: {llm_log_file}")

    loop = InsightLoop(
        goal=args.goal,
        rag_store=rag_store,
        llm=llm,
        max_iterations=insight_max_iters,
        top_k=args.top_k,
        queries_per_iteration=args.queries_per_iteration,
        max_evidence=args.max_evidence,
        target_opportunities=args.target_opportunities,
        goal_description=args.goal_description,
        use_agent=getattr(args, "insight_agent", False),
        llm_log_file=llm_log_file,
        **autonomous_kwargs,
    )
    opportunities = loop.run()

    from token_tracker import get_tracker as _get_tracker
    report = generate_report(
        opportunities, goal=args.goal, queries_explored=loop.all_queries,
        token_usage=_get_tracker().to_dict(),
    )

    with open(args.out_report, "w") as f:
        f.write(report)
    print(f"\nReport written to {args.out_report}")


def _connect_pg(args: argparse.Namespace):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(args.pg_url)
    register_vector(conn)
    print(f"Connected to PostgreSQL at {args.pg_url}")
    return conn


def _should_run(step: str, start_from: str) -> bool:
    return STEPS.index(step) >= STEPS.index(start_from)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

    _finalise_run_id(args)

    import logging
    if args.verbose:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        for name in ("research_loop", "research_loop_openai", "insights"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        # OpenAI Agents SDK logs (agent turns, tool calls)
        if args.openai_agent:
            logging.getLogger("openai.agents").setLevel(logging.INFO)

    print(f"Pipeline run ID: {args.run_id}")
    print(f"  Directory: {args.run_dir}")
    print(f"  Corpus: {args.out_corpus}")
    print(f"  Report: {args.out_report}")
    print(f"  Namespace: {args.namespace}")
    _write_params(args, argv)
    print()

    if args.start_from != "research":
        print(f"Resuming pipeline from step: {args.start_from}")

    # Step 1: Research
    if _should_run("research", args.start_from):
        _banner("Step 1/3: Research Loop")
        try:
            total_docs = _run_research(args)
        except Exception as exc:
            _print_failure("research", exc, args)
            sys.exit(1)

        if args.dry_run:
            print("\nDry-run mode: stopping after research loop.")
            return

        if total_docs == 0:
            print("\nResearch loop produced 0 documents. Aborting pipeline.")
            sys.exit(1)
    else:
        # Skipping research — validate corpus file exists
        if not os.path.isfile(args.out_corpus):
            print(f"Error: corpus file not found: {args.out_corpus}", file=sys.stderr)
            print("Cannot resume without the research corpus. Run the full pipeline or check --out-corpus path.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Using existing corpus: {args.out_corpus}")

    if args.dry_run:
        print("\nDry-run mode: stopping before DB steps.")
        return

    # Step 2: RAG Indexing
    conn = None
    if _should_run("index", args.start_from):
        _banner("Step 2/3: RAG Indexing")
        try:
            conn = _connect_pg(args)
        except Exception as exc:
            _print_failure("index", exc, args)
            sys.exit(1)

        try:
            indexed = _run_index(args, conn)
        except Exception as exc:
            conn.close()
            _print_failure("index", exc, args)
            sys.exit(1)

        if indexed == 0:
            print("\nNo vectors indexed. Aborting pipeline.")
            conn.close()
            sys.exit(1)

    # Step 3: Insights
    _banner("Step 3/3: Insight Discovery")
    try:
        if conn is None:
            conn = _connect_pg(args)
        _run_insights(args, conn)
    except Exception as exc:
        _print_failure("insights", exc, args)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()

    from token_tracker import get_tracker

    tracker = get_tracker()
    if tracker.has_usage():
        print(f"\n{tracker.summary()}")


if __name__ == "__main__":
    main()
