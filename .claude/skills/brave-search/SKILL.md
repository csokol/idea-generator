---
name: brave-search
description: Reference documentation for the Brave Search APIs (Web Search and LLM Context). Load this skill when making changes to code that integrates with the Brave Search API.
user-invocable: true
disable-model-invocation: false
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Brave Search API Reference

Brave offers two search endpoints sharing the same authentication and error codes:

| API | Endpoint | Best for |
|-----|----------|----------|
| **Web Search** | `/res/v1/web/search` | Standard search results with rich metadata |
| **LLM Context** | `/res/v1/llm/context` | Pre-extracted content for RAG / LLM grounding |

## Authentication (both APIs)

Header: `X-Subscription-Token: <API_KEY>`

## Error Codes (both APIs)

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Bad request — parameter invalid or not in plan |
| 403 | Forbidden — resource not allowed for subscription |
| 404 | Not found — subscription missing |
| 422 | Invalid subscription token |
| 429 | Rate limited or quota exceeded |

---

# Web Search API

## Endpoint

```
GET https://api.search.brave.com/res/v1/web/search
```

Standard web search returning structured results with titles, descriptions, snippets, and rich metadata.

## Required Parameters

| Parameter | Type | Constraints |
|-----------|------|-------------|
| `q` | string | Supports search operators (`"quotes"`, `-minus`, `site:`, `filetype:`) |

## Optional Query Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `count` | integer | 20 | max 20 | Results per page |
| `offset` | integer | 0 | max 9 | Pagination offset (0-based) |
| `freshness` | string | — | `pd`/`pw`/`pm`/`py` or custom range | Recency filter (`pd`=past day, `pw`=past week, `pm`=past month, `py`=past year) |
| `country` | string | — | 2-char code | Target region |
| `search_lang` | string | — | ISO 639-1 | Content language filter |
| `ui_lang` | string | — | e.g. `en-US` | Language for response metadata |
| `extra_snippets` | boolean | false | — | Return up to 5 additional excerpts per result |
| `safesearch` | string | moderate | `off`/`moderate`/`strict` | Adult content filtering |
| `goggles` | string | — | URL or inline definition | Custom re-ranking rules |

## Response Structure

```json
{
  "query": {
    "original": "string",
    "more_results_available": true
  },
  "web": {
    "results": [
      {
        "title": "string",
        "url": "string",
        "description": "string",
        "extra_snippets": ["string"],
        "age": "string",
        "language": "string",
        "family_friendly": true,
        "page_age": "string",
        "page_fetched": "string",
        "thumbnail": { "src": "string" },
        "deep_results": {},
        "schemas": []
      }
    ]
  },
  "news": { "results": [] },
  "videos": { "results": [] },
  "discussions": { "results": [] },
  "infobox": {},
  "mixed": {}
}
```

### Key Response Fields

- **`query`** — Echo of the query with `original` text and `more_results_available` boolean.
- **`web.results[]`** — Main web results:
  - `title` — Page heading
  - `url` — Page URL
  - `description` — Page summary / snippet
  - `extra_snippets` — Additional excerpts (only if `extra_snippets=true` in request)
  - `age` — Human-readable recency (e.g. "2 hours ago")
  - `language` — Content language code
  - `page_age` — Publication date
  - `page_fetched` — Last crawl timestamp
  - `thumbnail` — Image thumbnail object with `src`
  - `deep_results` — Sub-page links and buttons
  - `schemas` — Extracted schema.org structured data
- **`news.results[]`** — News articles (optional)
- **`videos.results[]`** — Video results (optional)
- **`discussions.results[]`** — Forum/discussion results (optional)
- **`infobox`** — Knowledge graph entity card (optional)
- **`mixed`** — Ranking order of result types on the page

## Example Request (cURL)

```bash
curl "https://api.search.brave.com/res/v1/web/search?q=how+deep+is+the+mediterranean+sea&count=5&extra_snippets=true" \
  -H "Accept: application/json" \
  -H "X-Subscription-Token: $BRAVE_SEARCH_TOKEN"
```

## Example Request (Python / httpx)

```python
import httpx

resp = httpx.get(
    "https://api.search.brave.com/res/v1/web/search",
    params={"q": "how deep is the mediterranean sea", "count": 5, "extra_snippets": True},
    headers={
        "Accept": "application/json",
        "X-Subscription-Token": token,
    },
)
resp.raise_for_status()
data = resp.json()

for r in data.get("web", {}).get("results", []):
    print(r["title"], r["url"])
    print(r.get("description", ""))
    for s in r.get("extra_snippets", []):
        print(" ", s)
```

---

# LLM Context API

## Endpoint

```
GET https://api.search.brave.com/res/v1/llm/context
```

Pre-extracted web content optimized for AI agents, LLM grounding, and RAG pipelines.

## Required Parameters

| Parameter | Type | Constraints |
|-----------|------|-------------|
| `q` | string | 1-400 characters, max 50 words |

## Optional Query Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `count` | integer | 20 | 1-50 | Number of results considered |
| `country` | string | US | 2-char code | Country for results |
| `search_lang` | string | en | 2+ char code | Search language |
| `maximum_number_of_urls` | integer | 20 | 1-50 | Max URLs in response |
| `maximum_number_of_tokens` | integer | 8192 | 1024-32768 | Max total tokens |
| `maximum_number_of_snippets` | integer | 50 | 1-100 | Max total snippets |
| `context_threshold_mode` | string | balanced | disabled/strict/lenient/balanced | Filtering aggressiveness |
| `maximum_number_of_tokens_per_url` | integer | 4096 | 512-8192 | Max tokens per individual URL |
| `maximum_number_of_snippets_per_url` | integer | 50 | 1-100 | Max snippets per individual URL |
| `goggles` | string | null | URL or inline definition | Reranking rules |
| `enable_local` | boolean | auto-detect | | Toggle local recall |

## Optional Headers

| Header | Type | Description |
|--------|------|-------------|
| `x-loc-lat` | number | Latitude (-90 to +90) |
| `x-loc-long` | number | Longitude (-180 to +180) |
| `x-loc-city` | string | Client city |
| `x-loc-state` | string | State/region code |
| `x-loc-state-name` | string | State/region name |
| `x-loc-country` | string | 2-letter country code |
| `x-loc-postal-code` | string | Postal code |
| `api-version` | string | YYYY-MM-DD format |
| `cache-control` | string | "no-cache" to bypass cache |
| `user-agent` | string | Browser user-agent string |

## Response Structure

```json
{
  "grounding": {
    "generic": [
      {
        "url": "string",
        "title": "string",
        "snippets": ["string"]
      }
    ],
    "poi": {
      "name": "string",
      "url": "string",
      "title": "string",
      "snippets": ["string"]
    },
    "map": [
      {
        "name": "string",
        "url": "string",
        "title": "string",
        "snippets": ["string"]
      }
    ]
  },
  "sources": {
    "<url>": {
      "title": "string",
      "hostname": "string",
      "age": ["string"]
    }
  }
}
```

### Response Fields

- **`grounding.generic[]`** — Main web results. Each has `url`, `title`, and `snippets` (array of text strings).
- **`grounding.poi`** — Point-of-interest result (single object, optional). Has `name`, `url`, `title`, `snippets`.
- **`grounding.map[]`** — Map/location results (optional). Same shape as `poi`.
- **`sources`** — Metadata keyed by URL. Each entry has `title`, `hostname`, and `age` (array of age strings).

## Example Request (cURL)

```bash
curl "https://api.search.brave.com/res/v1/llm/context?q=how+deep+is+the+mediterranean+sea" \
  -H "Accept: application/json" \
  -H "X-Subscription-Token: $BRAVE_SEARCH_TOKEN"
```

## Example Request (Python / httpx)

```python
import httpx

resp = httpx.get(
    "https://api.search.brave.com/res/v1/llm/context",
    params={"q": "how deep is the mediterranean sea"},
    headers={
        "Accept": "application/json",
        "X-Subscription-Token": token,
    },
)
resp.raise_for_status()
data = resp.json()

# Main results
for item in data.get("grounding", {}).get("generic", []):
    print(item["title"], item["url"])
    for snippet in item["snippets"]:
        print(" ", snippet)
```
