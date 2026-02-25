# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python MVP project. Uses modern Python packaging (`pyproject.toml` + hatchling) with Python 3.14+ and `uv` for dependency/project management.

Contains a **Brave Search CLI tool**, an **LLM agent**, a **Research Loop** for iterative LLM-guided corpus building, an **OpenAI Agents Research Loop** (parallel implementation using OpenAI Agents SDK + WebSearchTool), a **RAG Indexer** for embedding and indexing research corpus into PostgreSQL + pgvector, and an **Insight Agent** that queries the RAG store to discover and rank business opportunities.

## Commands

- **Run Brave Search**: `uv run brave-search "your query"`
- **Run Agent**: `uv run agent "your question"` (or omit query for interactive REPL)
- **Run Full Pipeline**: `uv run pipeline --goal "your goal" --domains example.com` — runs research → RAG index → insights end-to-end. Auto-generates run ID and outputs under `data/<run-id>/`. Supports `--start-from {research,index,insights}` to resume from a failed step, and `--run-id` to reuse an existing run directory. Key flags: `--freshness`, `--fetch-pages`, `--iterations`, `--openai-agent`, `--insight-agent`, `--insight-provider`, `--dry-run`, `--verbose`.
- **Run Research Loop**: `uv run research-loop --goal "your goal" --domains example.com --iterations 3`
- **Run Research Loop (OpenAI)**: `uv run research-loop-openai --goal "your goal" --domains example.com --iterations 3`
- **Run RAG Indexer**: `uv run index-rag --in corpus.jsonl --index myindex --namespace dev` (add `--dry-run` to skip DB)
- **Run Insight Agent**: `uv run insights --goal "your goal" --namespace dev --index myindex` (add `--dry-run` to verify DB connection only; add `--agent` to use autonomous LangGraph ReAct agent mode)
- **Run Webapp**: `uv run webapp` (browse results at http://127.0.0.1:8000; `--data-dir`, `--host`, `--port`)
- **Start PostgreSQL**: `docker compose up -d`
- **Run tests**: `make test` or `uv run pytest`
- **Sync deps**: `uv sync`
- **Add dependency**: `uv add <package>` (or `uv add --dev <package>` for dev deps)

## Architecture

### Brave Search CLI (`brave_search/`)

- **`brave_search/cli.py`** — CLI entrypoint and all logic
  - `get_token()` — reads `BRAVE_SEARCH_TOKEN` from env
  - `build_params()` — constructs API query params from CLI args
  - `fetch_results()` — calls Brave API via `httpx`
  - `format_output()` — formats results (human-readable, JSON, or raw)
  - `main()` — argparse-based CLI entrypoint
- **`brave_search/storage.py`** — result persistence and search logging
  - `ResultStore(data_dir)` — class managing result files and search log under a data directory
    - `.save_result(data, query, *, now)` — write JSON result file
    - `.append_log(query, params, result_count, result_file, *, now)` — append to JSONL search log
  - Module-level `save_result()` and `append_log()` shims delegate to `ResultStore` for backward compatibility
  - Private helpers `_ensure_dirs()`, `_slugify()`, `_timestamp_str()` remain module-level (pure)
- **Entrypoint**: registered in `pyproject.toml` as `brave-search = "brave_search.cli:main"`
- **Dependencies**: `httpx` for HTTP requests
- **API**: `GET https://api.search.brave.com/res/v1/llm/context` (LLM Context API, no freshness/date filter support) with `X-Subscription-Token` header
- **CLI args**: `--count`, `--country`, `--search-lang`, `--max-urls`, `--max-tokens`, `--max-snippets`, `--threshold-mode`, `--json`, `--raw`
- **Note**: The research loop (`research_loop/search.py`) uses the separate **Web Search API** (`/res/v1/web/search`) which supports `freshness` date filtering

### Research Loop (`research_loop/`)

- **`research_loop/cli.py`** — CLI entrypoint with argparse
- **`research_loop/loop.py`** — core iteration orchestration
  - `ResearchLoop(**config)` — class encapsulating loop state (run_id, dedup, total_docs, all_queries)
    - `.run()` — execute the full research loop
    - `._run_iteration()`, `._process_query()`, `._process_result()` — iteration internals
  - Module-level `run_loop(**kwargs)` shim delegates to `ResearchLoop` for backward compatibility
  - Uses module-level `brave_search()` import so monkeypatching in tests still works
- **`research_loop/query_gen.py`** — LLM query generation (provider-agnostic)
  - `QueryGenerator(llm, goal, domains)` — class binding LLM + goal + domains
    - `.generate(num_queries, iteration, previous_queries)` — generate search queries via LLM
  - Module-level `generate_queries()` shim delegates to `QueryGenerator`
  - `_fallback_queries()` stays module-level (pure helper)
- **`research_loop/search.py`** — Brave Web Search API wrapper (`/res/v1/web/search`, NOT the LLM Context API)
  - `BraveSearchClient(token, timeout, api_url)` — class wrapping API config
    - `.search(query, count, freshness)` — execute search (raises `RuntimeError` on failure). `freshness` filters by age: `pd` (24h), `pw` (7d), `pm` (31d), `py` (1y), or date range `YYYY-MM-DDtoYYYY-MM-DD`
    - `.extract_results(data)` — static method to extract web results from `data["web"]["results"]`
  - Module-level `brave_search()` and `extract_results()` shims for backward compatibility
- **`research_loop/fetch.py`** — optional page fetching (HTML → text via BeautifulSoup)
- **`research_loop/records.py`** — JSONL record building, URL dedup, file writing
- **Entrypoint**: `research-loop = "research_loop.cli:main"`
- **Dependencies**: `httpx`, `langchain-groq`, `langchain-google-genai`, `beautifulsoup4`
- **CLI args**: `--goal`, `--domains`, `--iterations`, `--per-iter-queries`, `--results-per-query`, `--max-docs`, `--out`, `--fetch-pages`, `--freshness`, `--timeout-sec`, `--sleep-sec`, `--user-agent`, `--dry-run`, `--provider`, `--model`, `--verbose`

### Pipeline (`pipeline/`)

- **`pipeline/cli.py`** — end-to-end CLI: research → RAG index → insights in one command
  - `parse_args()` — unified argparse with argument groups for each step
  - `main()` — orchestrates all three steps sequentially, with error recovery and resume support
  - `_finalise_run_id()` — generates human-readable run ID: `<timestamp>_<llm-slug>_<model-tag>` (uses `slug.py`)
  - `_run_research()` — dispatches to Brave or OpenAI research loop based on `--openai-agent`
  - `_run_index()` — reads corpus JSONL, chunks, embeds, upserts into pgvector
  - `_run_insights()` — runs insight loop and writes markdown report
  - `_write_params()` — persists run params to `data/<run-id>/params.json` (used by webapp)
  - `_build_resume_cmd()` — on failure, prints the exact command to resume from the failed step
- **Entrypoint**: `pipeline = "pipeline.cli:main"`
- **Output structure**: `data/<run-id>/params.json`, `data/<run-id>/corpus.jsonl`, `data/<run-id>/report.md`
- **Key CLI args**:
  - Shared: `--goal`, `--domains`, `--run-id`, `--data-dir`, `--start-from {research,index,insights}`, `--dry-run`, `--verbose`
  - Research: `--iterations`, `--per-iter-queries`, `--results-per-query`, `--max-docs`, `--fetch-pages`, `--freshness`, `--openai-agent`, `--openai-model`, `--research-provider`, `--research-model`
  - RAG indexer: `--pg-url`, `--index`, `--namespace`, `--embed-provider` (default `openai`), `--embed-model`, `--chunk-size`, `--chunk-overlap`
  - Insights: `--insight-max-iterations` (alias `--insight-iterations`), `--target-opportunities`, `--goal-description`, `--insight-agent` (use autonomous LangGraph ReAct agent), `--insight-provider`, `--insight-model`, `--top-k`, `--queries-per-iteration`, `--max-evidence`

### OpenAI Agents Research Loop (`research_loop_openai/`)

- **`research_loop_openai/agent.py`** — OpenAI Agent + structured output
  - `SearchResult` (Pydantic): url, title, snippet, source_domain
  - `IterationOutput` (Pydantic): results list, summary, suggested_next_queries
  - `ResearchAgent(goal, domains, model)` — wraps OpenAI Agent with WebSearchTool
    - `.run_iteration(iteration, num_queries, previous_queries, research_summary, known_urls)` — async, returns `IterationOutput`
    - `.run_iteration_sync(...)` — sync wrapper via `asyncio.run()`
- **`research_loop_openai/loop.py`** — orchestrator (same shape as existing `ResearchLoop`)
  - `OpenAIResearchLoop(**config)` — class encapsulating loop state
    - `.run()` — execute the full research loop
    - Reuses `research_loop.records` (Deduplicator, build_record, append_jsonl) and `research_loop.fetch` (fetch_page)
    - Same JSONL output format, same dedup logic, same stopping conditions (max_docs, no new docs)
  - Module-level `run_loop(**kwargs)` shim
- **`research_loop_openai/cli.py`** — CLI entrypoint with argparse
- **Entrypoint**: `research-loop-openai = "research_loop_openai.cli:main"`
- **Dependencies**: `openai-agents` (provides `agents` package with Agent, Runner, WebSearchTool)
- **CLI args**: `--goal`, `--domains`, `--iterations`, `--per-iter-queries`, `--results-per-query`, `--max-docs`, `--out`, `--fetch-pages`, `--timeout-sec`, `--sleep-sec`, `--user-agent`, `--dry-run`, `--openai-model`

### Agent (`agent/`)

- **`agent/cli.py`** — CLI entrypoint with argparse (single-shot or interactive REPL)
- **`agent/graph.py`** — LangGraph ReAct agent construction (`build_agent()`, `run_agent()`)
- **`agent/tools.py`** — Brave Search tool definition for the agent
- **Entrypoint**: `agent = "agent.cli:main"`
- **Dependencies**: `langgraph`, `llm_factory`
- **CLI args**: `query` (positional, optional), `--provider`, `--model`

### RAG Indexer (`index_rag/`)

- **`index_rag/cli.py`** — CLI entrypoint with argparse
- **`index_rag/reader.py`** — JSONL reading, filtering (dedup, min text length), text extraction
- **`index_rag/chunker.py`** — character-based text chunking with overlap on whitespace boundaries
- **`index_rag/embedder.py`** — embedding wrapper with pluggable providers: `local` (sentence-transformers, 384d), `google` (Gemini `gemini-embedding-001`, 768d), `openai` (text-embedding-3-small, 1536d); default provider is `openai`
- **`index_rag/indexer.py`** — PostgreSQL + pgvector upsert orchestration
  - `PgVectorIndexer(conn, table_name, dimension)` — class wrapping connection + table config
    - `.ensure_index()` — create extension, table, and indexes
    - `.upsert_batched(vectors, namespace, batch_size)` — batched upsert with ON CONFLICT
  - Module-level `ensure_index()` and `upsert_batched()` shims for backward compatibility
  - `build_vector()` and `_truncate()` stay module-level (pure, stateless)
- **`index_rag/ids.py`** — stable doc/chunk ID generation (SHA256 of canonical URL + chunk index)
- **Entrypoint**: `index-rag = "index_rag.cli:main"`
- **Dependencies**: `psycopg[binary]`, `pgvector`, `sentence-transformers`
- **CLI args**: `--in`, `--index`, `--namespace`, `--pg-url`, `--dimension`, `--batch-size`, `--chunk-size`, `--chunk-overlap`, `--max-chunks-per-doc`, `--min-text-len`, `--model`, `--embed-provider`, `--dry-run`
- **Defaults**: `--embed-provider openai`, `--dimension 1536`, `--chunk-size 900`, `--chunk-overlap 120`, `--batch-size 100`, `--min-text-len 200`, `--max-chunks-per-doc 50`
- **Default PG URL**: `postgresql://postgres:postgres@localhost:5433/rag`
- **Infrastructure**: `docker-compose.yml` runs `pgvector/pgvector:pg17` on port 5433

### Insight Agent (`insights/`)

- **`insights/cli.py`** — CLI entrypoint with argparse
- **`insights/models.py`** — data classes for the insight pipeline
  - `Signal` dataclass: pain, workaround, desired_outcome, segment, severity, willingness_to_pay, keywords, chunk_id, url
  - `Opportunity` dataclass: title, summary, signals list, score, confidence; computed properties for evidence_count, unique_urls, avg_severity, avg_willingness_to_pay
  - `EvidenceStore` class: in-memory store tracking signals keyed by normalized pain, with dedup by chunk_id
- **`insights/rag_store.py`** — RAG store query layer over pgvector
  - `RagStore(conn, table_name, namespace, embedder)` — class wrapping DB connection + embedder
    - `.search(query_text, top_k, domains)` — semantic search with cosine similarity, optional domain filtering, embedding + result caching
    - `.get_doc(doc_id)` — retrieve all chunks for a document by doc_id prefix
- **`insights/tools.py`** — LangChain tool wrappers (factory functions)
  - `make_rag_search_tool(rag_store)` → `@tool rag_search`
  - `make_rag_get_doc_tool(rag_store)` → `@tool rag_get_doc`
- **`insights/prompts.py`** — all LLM prompt templates as string constants
  - `SYSTEM_PROMPT`, `PLAN_QUERIES_PROMPT`, `EXTRACT_SIGNALS_PROMPT`, `CLUSTER_MERGE_PROMPT`, `EVALUATE_PROGRESS_PROMPT`
- **`insights/loop.py`** — insight agent orchestrator
  - `InsightLoop(goal, rag_store, llm, *, max_iterations, top_k, queries_per_iteration, max_evidence, min_evidence_per_opportunity, target_opportunities, goal_description)` — iterative loop
    - `.run()` → list of Opportunity objects
    - `._plan_queries()` — LLM generates diverse search queries per iteration
    - `._retrieve()` — search RAG store for each query, expand chunk hits to full documents
    - `._extract_signals()` — LLM extracts structured signals from full page text (batched by 3 docs)
    - `._evaluate_progress()` — agentic stop/continue decision after each iteration (with `target_opportunities` structured override)
    - `._cluster_and_merge()` — LLM merges similar pain groups into opportunity clusters
    - `._score_opportunities()` — rank by weighted evidence count, severity, WTP, source diversity
    - Stop conditions: `max_evidence` reached, no new docs in last iteration, `max_iterations` exhausted, agentic evaluation stops, `target_opportunities` reached
    - Rate-limit retry: exponential backoff up to 5 attempts on 429 / RESOURCE_EXHAUSTED errors
- **`insights/report.py`** — markdown report generation
  - `generate_report(opportunities, goal, queries_explored)` → markdown string with opportunity briefs, evidence citations, metrics, and appendix
- **Entrypoint**: `insights = "insights.cli:main"`
- **Dependencies**: reuses `index_rag.embedder`, `llm_factory`, `psycopg`, `pgvector`
- **CLI args**: `--goal`, `--namespace`, `--domains`, `--top-k`, `--max-iterations` (with `--iterations` as alias), `--target-opportunities`, `--goal-description`, `--queries-per-iteration`, `--max-evidence`, `--min-evidence-per-opportunity`, `--agent` (use autonomous LangGraph ReAct agent), `--out`, `--format`, `--pg-url`, `--index`, `--embed-provider`, `--embed-model`, `--provider`, `--model`, `--temperature`, `--verbose`, `--debug`, `--dry-run`
- **Defaults**: `--embed-provider openai`, `--provider gemini`, `--max-iterations 3`, `--top-k 10`, `--queries-per-iteration 5`, `--max-evidence 200`, `--min-evidence-per-opportunity 2`

### LLM Factory (`llm_factory.py`)

- **`llm_factory.py`** — shared factory for pluggable LLM providers
  - `build_llm(provider, model)` — returns a `BaseChatModel` for `"groq"`, `"gemini"`, or `"openai"`
  - `PROVIDERS` dict — maps provider names to env var and default model
  - Providers and default models:
    - `groq` (default): `qwen/qwen3-32b`, uses `GROQ_API_KEY`
    - `gemini`: `gemini-3-flash-preview`, uses `GOOGLE_API_KEY`
    - `openai`: `gpt-4.1`, uses `OPENAI_API_KEY`

### Slug Generator (`slug.py`)

- **`slug.py`** — generates human-friendly run ID slugs via LLM
  - `generate_slug(goal, llm)` — ask LLM for a short ≤10-char filesystem-safe slug; falls back to sanitized goal
  - `generate_slug_openai(goal, model)` — same using OpenAI SDK directly (no LangChain)
  - `model_tag(provider, model)` — returns a ≤20-char tag like `groq-qwen3-32b` for use in run IDs
  - `_fallback_slug(goal)` — derive slug from goal without LLM (pure)

### Webapp (`webapp/`)

- **`webapp/data.py`** — data loading for pipeline runs
  - `RunSummary` dataclass: run_id, goal, started_at, domains, corpus_count, has_report
  - `RunDetail` dataclass: run_id, params dict, report_html, corpus_count
  - `list_runs(data_dir)` — scan `data/*/params.json`, sort by date desc
  - `get_run(run_id, data_dir)` — load params, render `report.md` → HTML via `markdown`
- **`webapp/app.py`** — FastAPI app + CLI
  - `GET /` — index page listing all runs
  - `GET /run/{run_id}` — detail page with metadata + rendered report
  - `main()` — argparse CLI (`--data-dir`, `--host`, `--port`) → `uvicorn.run()`
- **`webapp/templates/`** — Jinja2 templates with Pico CSS v2 (CDN)
- **Entrypoint**: `webapp = "webapp.app:main"`
- **Dependencies**: `fastapi[standard]`, `markdown`

### Other

- **`main.py`** — original placeholder entrypoint

### Tests (`tests/`)

- **`tests/test_cli.py`** — Brave Search CLI unit tests (monkeypatched `httpx`, no network calls)
- **`tests/test_agent_e2e.py`** — agent e2e tests with `FakeChatModel` (tool calls, error handling, multi-turn)
- **`tests/test_agent_tools.py`** — agent tool unit tests
- **`tests/test_research_loop.py`** — research loop tests (records, dedup, query gen, search, loop integration)
- **`tests/test_index_rag.py`** — RAG indexer tests (reader, chunker, IDs, dry-run pipeline; mocked embeddings/PostgreSQL)
- **`tests/test_rag_e2e.py`** — integration tests: research loop → RAG indexer → real PostgreSQL + pgvector (requires Docker; `@pytest.mark.integration`)
- **`tests/test_insights.py`** — Insight agent tests (models, EvidenceStore dedup, RagStore SQL/caching, report generation, full loop integration with mocked RAG+LLM, tool wrappers)
- **`tests/test_storage.py`** — Brave Search storage/logging tests
- Dev dependency: `pytest`
- **Marker**: `integration` — tests requiring external services (e.g. PostgreSQL). Run `uv run pytest -m "not integration"` to skip them

## Environment

- Requires `BRAVE_SEARCH_TOKEN` env var set to a valid Brave Search API key
- Requires `GROQ_API_KEY` env var for LLM features with Groq provider (default)
- Requires `GOOGLE_API_KEY` env var for LLM features with Gemini provider (`--provider gemini`) and for the Google embedding provider (`--embed-provider google`)
- Requires `OPENAI_API_KEY` env var for the OpenAI Agents research loop (`research-loop-openai`), the `openai` LLM provider, and the default embedding provider (`--embed-provider openai`)
