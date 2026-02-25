# ideas-generator

## Brave Search CLI

A CLI tool to query the [Brave LLM Context API](https://brave.com/search/api/).

### Setup

```bash
cp .env.example .env
# Edit .env with your API keys
uv sync
```

### Usage

```bash
# Basic search
uv run brave-search "hello world"

# Limit results
uv run brave-search "shipping rules" --count 5

# JSON output
uv run brave-search "shipping rules" --count 5 --json

# Full raw API response
uv run brave-search "site:community.shopify.com returns app" --raw

# With filters
uv run brave-search "restaurants" --country US --search-lang en

# Skip saving results to disk
uv run brave-search "test query" --no-save
```

### Options

| Flag                  | Description                                              | Default    |
|-----------------------|----------------------------------------------------------|------------|
| `--count N`           | Number of results (max 50)                               | 20         |
| `--country CC`        | Country code filter                                      |            |
| `--search-lang`       | Search language                                          |            |
| `--max-urls N`        | Maximum number of URLs (1–50)                            |            |
| `--max-tokens N`      | Maximum number of tokens (1024–32768)                    |            |
| `--max-snippets N`    | Maximum number of snippets (1–100)                       |            |
| `--threshold-mode`    | Context threshold mode (`strict`, `balanced`, `lenient`, `disabled`) | |
| `--json`              | Output as compact JSON array                             |            |
| `--raw`               | Output full API response                                 |            |
| `--no-save`           | Do not save results or log the search                    |            |

## AI Agent

An AI agent powered by LangGraph that can answer questions using Brave Search as a tool. Supports pluggable LLM providers (Groq, Gemini, and OpenAI).

### Setup

Requires `BRAVE_SEARCH_TOKEN` and at least one LLM API key in your `.env` file:
- `GROQ_API_KEY` for Groq (default)
- `GOOGLE_API_KEY` for Gemini
- `OPENAI_API_KEY` for OpenAI

### Usage

```bash
# Single-shot: ask a question and get an answer
uv run agent "What happened in the news today?"

# Use Gemini instead of Groq
uv run agent --provider gemini "What is the capital of France?"

# Use OpenAI
uv run agent --provider openai "Quick question"

# Use a specific model
uv run agent --provider groq --model llama-3.1-8b-instant "Quick question"

# Interactive REPL: multi-turn conversation
uv run agent
# > What are the latest Python releases?
# > Tell me more about the new features
# > quit
```

## Research Loop

An iterative LLM-guided research tool that builds a corpus of documents. It generates search queries via an LLM, executes them against Brave Search, optionally fetches page content, and logs all discovered documents as JSONL records.

### Setup

Requires `BRAVE_SEARCH_TOKEN` and at least one LLM API key in your `.env` file:
- `GROQ_API_KEY` for Groq (default)
- `GOOGLE_API_KEY` for Gemini
- `OPENAI_API_KEY` for OpenAI

### Usage

```bash
# Basic research loop — generates queries, searches, and writes JSONL
uv run research-loop \
  --goal "find opportunities for Shopify apps" \
  --domains community.shopify.com apps.shopify.com

# Control iteration count and output limits
uv run research-loop \
  --goal "pain points with returns management" \
  --domains community.shopify.com \
  --iterations 3 \
  --per-iter-queries 4 \
  --results-per-query 10 \
  --max-docs 50 \
  --out shopify-returns.jsonl

# Fetch full page content (HTML → text) for each result
uv run research-loop \
  --goal "feature requests for inventory apps" \
  --domains apps.shopify.com community.shopify.com \
  --fetch-pages \
  --out inventory-research.jsonl

# Filter by freshness (recent results only)
uv run research-loop \
  --goal "latest Shopify app reviews" \
  --domains community.shopify.com \
  --freshness pw

# Dry run — generates queries and searches, but skips page fetching
uv run research-loop \
  --goal "test query generation" \
  --domains example.com \
  --iterations 1 \
  --dry-run

# Use Gemini instead of Groq
uv run research-loop \
  --goal "competitor analysis" \
  --domains reddit.com \
  --provider gemini

# Use a specific model
uv run research-loop \
  --goal "competitor analysis" \
  --domains reddit.com \
  --provider groq --model llama-3.1-8b-instant
```

### Options

| Flag                   | Description                                                                        | Default                              |
|------------------------|------------------------------------------------------------------------------------|--------------------------------------|
| `--goal`               | Research goal description (required)                                               |                                      |
| `--domains`            | Target domains to search (required)                                                |                                      |
| `--iterations N`       | Number of research iterations                                                      | 5                                    |
| `--per-iter-queries N` | Queries generated per iteration                                                    | 5                                    |
| `--results-per-query N`| Search results per query                                                           | 5                                    |
| `--max-docs N`         | Stop after collecting N documents                                                  | 100                                  |
| `--out PATH`           | Output JSONL file path                                                             | `research.jsonl`                     |
| `--fetch-pages`        | Fetch and extract page content                                                     | off                                  |
| `--freshness`          | Filter by age: `pd` (24h), `pw` (7d), `pm` (31d), `py` (1y), or date range       |                                      |
| `--timeout-sec N`      | HTTP timeout in seconds                                                            | 15                                   |
| `--sleep-sec N`        | Sleep between requests (rate limiting)                                             | 0.5                                  |
| `--user-agent`         | User-Agent header for page fetching                                                | `ResearchLoop/0.1 (compatible; bot)` |
| `--dry-run`            | Search but skip page fetching                                                      | off                                  |
| `--provider`           | LLM provider (`groq`, `gemini`, or `openai`)                                       | `groq`                               |
| `--model`              | LLM model name                                                                     | depends on provider                  |

### Output Format

Each line in the output JSONL file is a JSON object with these fields:

```json
{
  "run_id": "20260214_153000_shopify_groq-qwen3-32b",
  "ts": "2026-02-14T15:30:01.123456+00:00",
  "goal": "find opportunities for Shopify apps",
  "domains": ["community.shopify.com"],
  "iteration": 1,
  "query": "site:community.shopify.com returns app pain points",
  "rank": 0,
  "url": "https://community.shopify.com/...",
  "title": "Need help with returns",
  "snippet": "I'm looking for a better returns solution...",
  "source_domain": "community.shopify.com",
  "content": null,
  "dedupe": "unique",
  "errors": []
}
```

## Research Loop (OpenAI Agents)

An alternative research loop implementation backed by the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Instead of generating queries via a LangChain LLM and searching via Brave API separately, this version uses an OpenAI Agent with the built-in `WebSearchTool` — the agent plans queries, executes web searches, and returns structured results in a single step.

Produces the **same JSONL output format** as the original research loop, so downstream tools (RAG indexer, insight agent, pipeline) work unchanged.

### Setup

Requires `OPENAI_API_KEY` in your `.env` file.

### Usage

```bash
# Basic research loop with OpenAI agent
uv run research-loop-openai \
  --goal "find opportunities for Shopify apps" \
  --domains community.shopify.com apps.shopify.com

# Control iterations and output
uv run research-loop-openai \
  --goal "pain points with returns management" \
  --domains community.shopify.com \
  --iterations 3 \
  --per-iter-queries 4 \
  --max-docs 50 \
  --out shopify-returns.jsonl

# Fetch full page content for each result
uv run research-loop-openai \
  --goal "feature requests for inventory apps" \
  --domains apps.shopify.com community.shopify.com \
  --fetch-pages \
  --out inventory-research.jsonl

# Use a specific OpenAI model
uv run research-loop-openai \
  --goal "competitor analysis" \
  --domains reddit.com \
  --openai-model gpt-4o-mini

# Dry run — agent searches but page fetching is skipped
uv run research-loop-openai \
  --goal "test" \
  --domains example.com \
  --iterations 1 \
  --dry-run
```

### Options

| Flag                   | Description                              | Default                              |
|------------------------|------------------------------------------|--------------------------------------|
| `--goal`               | Research goal description (required)     |                                      |
| `--domains`            | Target domains to search (required)      |                                      |
| `--iterations N`       | Number of research iterations            | 5                                    |
| `--per-iter-queries N` | Queries per iteration                    | 5                                    |
| `--results-per-query N`| Results per query                        | 5                                    |
| `--max-docs N`         | Stop after collecting N documents        | 100                                  |
| `--out PATH`           | Output JSONL file path                   | `research.jsonl`                     |
| `--fetch-pages`        | Fetch and extract page content           | off                                  |
| `--timeout-sec N`      | HTTP timeout in seconds                  | 15                                   |
| `--sleep-sec N`        | Sleep between requests (rate limiting)   | 0.5                                  |
| `--user-agent`         | User-Agent header for page fetching      | `ResearchLoop/0.1 (compatible; bot)` |
| `--dry-run`            | Search but skip page fetching            | off                                  |
| `--openai-model`       | OpenAI model name                        | `gpt-5-mini`                         |

## RAG Indexer

Embeds and indexes research corpus documents into PostgreSQL + pgvector for retrieval-augmented generation. Reads the JSONL output from the research loop, chunks the text, generates embeddings, and upserts vectors into a PostgreSQL table.

### Setup

Requires a running PostgreSQL instance with pgvector:

```bash
docker compose up -d
```

### Usage

```bash
# Index a research corpus into PostgreSQL (uses OpenAI embeddings by default)
uv run index-rag --in corpus.jsonl --index myindex --namespace dev

# Use local sentence-transformers model (no API key needed)
uv run index-rag --in corpus.jsonl --index myindex --namespace dev --embed-provider local

# Use Google embeddings
uv run index-rag --in corpus.jsonl --index myindex --namespace dev --embed-provider google

# Dry run — process and chunk documents without writing to the database
uv run index-rag --in corpus.jsonl --index myindex --namespace dev --dry-run

# Customize chunking and batching
uv run index-rag \
  --in inventory-research.jsonl \
  --index inventory \
  --namespace prod \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --batch-size 64
```

### Typical Workflow

1. Run the research loop to build a corpus:
   ```bash
   uv run research-loop --goal "your goal" --domains example.com --iterations 3 --out corpus.jsonl
   ```
2. Start PostgreSQL (if not already running):
   ```bash
   docker compose up -d
   ```
3. Index the corpus:
   ```bash
   uv run index-rag --in corpus.jsonl --index myindex --namespace dev
   ```

### Options

| Flag                     | Description                              | Default                                                 |
|--------------------------|------------------------------------------|---------------------------------------------------------|
| `--in`                   | Input JSONL file path (required)         |                                                         |
| `--index`                | Table name for vectors (required)        |                                                         |
| `--namespace`            | Namespace tag for vectors (required)     |                                                         |
| `--pg-url`               | PostgreSQL connection URL                | `postgresql://postgres:postgres@localhost:5433/rag`     |
| `--embed-provider`       | Embedding provider (`local`, `google`, `openai`) | `openai`                                        |
| `--dimension N`          | Embedding dimension (auto-set per provider) | 1536 (openai), 768 (google), 384 (local)             |
| `--batch-size N`         | Upsert batch size                        | 100                                                     |
| `--chunk-size N`         | Characters per chunk                     | 900                                                     |
| `--chunk-overlap N`      | Overlap between chunks                   | 120                                                     |
| `--max-chunks-per-doc N` | Max chunks per document                  | 50                                                      |
| `--min-text-len N`       | Minimum text length to index             | 200                                                     |
| `--model`                | Embedding model name override            | depends on provider                                     |
| `--dry-run`              | Process without writing to DB            | off                                                     |

## Insight Agent

Queries the RAG-indexed research corpus iteratively to discover, cluster, and rank business opportunities. Produces a markdown report with evidence citations.

### Setup

Requires:
- A running PostgreSQL instance with indexed data (see RAG Indexer above)
- An LLM API key (`GROQ_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`)

### Usage

```bash
# Discover opportunities from an indexed corpus
uv run insights \
  --goal "find opportunities for Shopify apps" \
  --namespace dev \
  --index myindex

# Filter to specific domains, more iterations
uv run insights \
  --goal "pain points with returns management" \
  --namespace dev \
  --index myindex \
  --domains community.shopify.com apps.shopify.com \
  --max-iterations 5 \
  --queries-per-iteration 8

# Stop once 10 opportunities are found
uv run insights \
  --goal "find opportunities" \
  --namespace dev \
  --index myindex \
  --target-opportunities 10

# Provide a richer goal description for the LLM
uv run insights \
  --goal "Shopify apps" \
  --goal-description "Focus on pain points for merchants with 10-100 orders per day" \
  --namespace dev \
  --index myindex

# Write report to file
uv run insights \
  --goal "inventory management opportunities" \
  --namespace dev \
  --index myindex \
  --out report.md

# Dry run — verify DB connection without running LLM
uv run insights --goal "test" --namespace dev --index myindex --dry-run

# Verbose progress output
uv run insights --goal "your goal" --namespace dev --index myindex --verbose

# Use Gemini instead of Groq
uv run insights \
  --goal "your goal" \
  --namespace dev \
  --index myindex \
  --provider gemini
```

### Options

| Flag                            | Description                                     | Default             |
|---------------------------------|-------------------------------------------------|---------------------|
| `--goal`                        | Research goal (required)                        |                     |
| `--namespace`                   | Namespace to search in                          | `""`                |
| `--domains`                     | Filter to specific source domains               | all                 |
| `--top-k N`                     | Results per RAG search query                    | 10                  |
| `--max-iterations N`            | Max search-extract iterations                   | 3                   |
| `--iterations N`                | Alias for `--max-iterations` (backward compat)  |                     |
| `--target-opportunities N`      | Stop when N qualified opportunities are found   |                     |
| `--goal-description`            | Richer goal description for the LLM agent       |                     |
| `--queries-per-iteration N`     | Queries generated per iteration                 | 5                   |
| `--max-evidence N`              | Stop after collecting N signals                 | 200                 |
| `--min-evidence-per-opportunity N` | Min signals to include an opportunity        | 2                   |
| `--out PATH`                    | Output file path (default: stdout)              |                     |
| `--format`                      | Output format                                   | `md`                |
| `--pg-url`                      | PostgreSQL connection URL                       | `postgresql://postgres:postgres@localhost:5433/rag` |
| `--index`                       | Table name                                      | `rag_chunks`        |
| `--embed-provider`              | Embedding provider (`local`, `google`, `openai`)| `openai`            |
| `--embed-model`                 | Embedding model name                            | depends on provider |
| `--provider`                    | LLM provider (`groq`, `gemini`, or `openai`)    | `gemini`            |
| `--model`                       | LLM model name                                  | depends on provider |
| `--temperature`                 | LLM temperature                                 | 0.3                 |
| `--verbose`                     | Structured progress logging                     | off                 |
| `--debug`                       | Debug logging with raw LLM prompts/responses    | off                 |
| `--dry-run`                     | Verify DB connection only                       | off                 |

### Typical End-to-End Workflow

1. Build a research corpus:
   ```bash
   uv run research-loop --goal "your goal" --domains example.com --iterations 3 --fetch-pages --out corpus.jsonl
   ```
2. Start PostgreSQL and index:
   ```bash
   docker compose up -d
   uv run index-rag --in corpus.jsonl --index myindex --namespace dev
   ```
3. Discover opportunities:
   ```bash
   uv run insights --goal "your goal" --namespace dev --index myindex --out report.md
   ```

## Pipeline (End-to-End)

Runs the full research → RAG indexing → insight discovery pipeline with a single command. Auto-generates a human-readable run ID (LLM slug + model tag + timestamp) and stores all outputs under `data/<run-id>/`. Supports resuming from any failed step.

### Usage

```bash
# Full pipeline: research, index, and generate insights report
uv run pipeline \
  --goal "find opportunities for Shopify apps" \
  --domains community.shopify.com apps.shopify.com \
  --fetch-pages

# Dry run: only run the research loop, skip DB and insights
uv run pipeline \
  --goal "pain points with returns management" \
  --domains community.shopify.com \
  --iterations 2 \
  --dry-run

# Resume from a failed step using an existing run ID
uv run pipeline \
  --goal "find opportunities for Shopify apps" \
  --domains community.shopify.com \
  --run-id 20260218_120000_shopify-groq-qwen3-32b \
  --start-from index

# Use OpenAI Agents SDK for research (instead of Brave Search + LangChain)
uv run pipeline \
  --goal "find opportunities for Shopify apps" \
  --domains community.shopify.com apps.shopify.com \
  --openai-agent \
  --fetch-pages

# OpenAI agent with a specific model
uv run pipeline \
  --goal "competitor analysis" \
  --domains community.shopify.com \
  --openai-agent \
  --openai-model gpt-4o \
  --iterations 3

# Stop insights once 10 qualified opportunities are found
uv run pipeline \
  --goal "find opportunities for Shopify apps" \
  --domains community.shopify.com \
  --target-opportunities 10

# Custom providers: Groq for research, Gemini for insights
uv run pipeline \
  --goal "inventory management opportunities" \
  --domains community.shopify.com apps.shopify.com \
  --research-provider groq \
  --insight-provider gemini \
  --fetch-pages

# Full control over all parameters
uv run pipeline \
  --goal "competitor analysis for shipping apps" \
  --domains community.shopify.com \
  --iterations 3 \
  --per-iter-queries 4 \
  --results-per-query 10 \
  --max-docs 50 \
  --fetch-pages \
  --freshness pw \
  --pg-url postgresql://postgres:postgres@localhost:5433/rag \
  --index shipping \
  --namespace v1 \
  --embed-provider openai \
  --insight-max-iterations 5 \
  --queries-per-iteration 8 \
  --max-evidence 300 \
  --verbose
```

### Output Structure

Each pipeline run creates a directory under `data/`:

```
data/<run-id>/
  params.json      — run configuration and metadata
  corpus.jsonl     — research corpus (JSONL records)
  report.md        — insight report (markdown)
```

### Options

| Flag                           | Description                                        | Default             |
|--------------------------------|----------------------------------------------------|---------------------|
| `--goal`                       | Research goal (required, used by all steps)        |                     |
| `--domains`                    | Target domains (required)                          |                     |
| `--run-id`                     | Reuse an existing run directory                    | auto-generated      |
| `--data-dir`                   | Base data directory                                | `data`              |
| `--start-from`                 | Resume from `research`, `index`, or `insights`     | `research`          |
| `--out-corpus PATH`            | JSONL corpus output path                           | `data/<run-id>/corpus.jsonl` |
| `--out-report PATH`            | Report output path                                 | `data/<run-id>/report.md`    |
| `--iterations N`               | Research loop iterations                           | 5                   |
| `--per-iter-queries N`         | Research queries per iteration                     | 5                   |
| `--results-per-query N`        | Search results per query                           | 5                   |
| `--max-docs N`                 | Max documents to collect                           | 100                 |
| `--fetch-pages`                | Fetch and extract page content                     | off                 |
| `--freshness`                  | Filter results by age (`pd`, `pw`, `pm`, `py`, or date range) |        |
| `--openai-agent`               | Use OpenAI Agents SDK research loop                | off                 |
| `--openai-model`               | OpenAI model when using `--openai-agent`           | `gpt-5-mini`        |
| `--research-provider`          | LLM provider for research                          | `groq`              |
| `--research-model`             | LLM model for research                             | depends on provider |
| `--pg-url`                     | PostgreSQL connection URL                          | `postgresql://postgres:postgres@localhost:5433/rag` |
| `--index`                      | Table name for vectors                             | `rag_chunks`        |
| `--namespace`                  | Namespace for vectors (default: run-id)            | run-id              |
| `--embed-provider`             | Embedding provider (`local`, `google`, `openai`)   | `openai`            |
| `--embed-model`                | Embedding model name                               | depends on provider |
| `--chunk-size N`               | Characters per chunk                               | 900                 |
| `--chunk-overlap N`            | Overlap between chunks                             | 120                 |
| `--insight-max-iterations N`   | Max insight loop iterations                        | 3                   |
| `--insight-iterations N`       | Alias for `--insight-max-iterations` (backward compat) |               |
| `--target-opportunities N`     | Stop insights when N qualified opportunities found |                     |
| `--goal-description`           | Richer goal description for the insight agent      |                     |
| `--insight-provider`           | LLM provider for insights                          | `gemini`            |
| `--insight-model`              | LLM model for insights                             | depends on provider |
| `--top-k N`                    | Results per RAG query                              | 10                  |
| `--queries-per-iteration N`    | Queries per insight iteration                      | 5                   |
| `--max-evidence N`             | Stop after N signals                               | 200                 |
| `--dry-run`                    | Run research only, skip DB + insights              | off                 |
| `--verbose`                    | Verbose progress output                            | off                 |

## Webapp

A FastAPI web application for browsing pipeline run results. Lists all runs and displays their configuration, corpus size, and rendered insight reports.

### Usage

```bash
# Start PostgreSQL (if not already running)
docker compose up -d

# Run the webapp
uv run webapp

# Browse at http://127.0.0.1:8000
```

### Options

| Flag          | Description                    | Default         |
|---------------|--------------------------------|-----------------|
| `--data-dir`  | Pipeline data directory        | `data`          |
| `--host`      | Host to bind to                | `127.0.0.1`     |
| `--port`      | Port to listen on              | `8000`          |

## Tests

```bash
make test
# or
uv run pytest

# Skip integration tests (require Docker/PostgreSQL)
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m integration
```
