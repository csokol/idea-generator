"""LangChain tools wrapping RagStore for the insight agent."""

from __future__ import annotations

from langchain_core.tools import tool

from insights.models import EvidenceStore, Signal
from insights.rag_store import RagStore


def make_rag_search_tool(rag_store: RagStore, top_k_default: int = 10):
    @tool
    def rag_search(query: str, top_k: int = top_k_default, domains: list[str] | None = None) -> str:
        """Search the RAG corpus for chunks semantically similar to the query.
        Returns ranked results with chunk text, URL, title, and relevance score.
        Use this to find evidence of user pain points, workarounds, and desired outcomes."""
        hits = rag_store.search(query, top_k=top_k, domains=domains)
        if not hits:
            return "No results found."
        parts = []
        for h in hits:
            parts.append(
                f"[{h['id']}] (score={h['score']:.3f}) {h['title']}\n"
                f"  URL: {h['url']}\n"
                f"  Domain: {h['source_domain']}\n"
                f"  Text: {h['chunk_text'][:500]}"
            )
        return "\n\n".join(parts)

    return rag_search


def make_rag_get_doc_tool(rag_store: RagStore):
    @tool
    def rag_get_doc(doc_id: str) -> str:
        """Retrieve all chunks for a specific document by its doc_id (the hash before the # in chunk IDs).
        Use this to get full context for a document when a chunk looks promising."""
        chunks = rag_store.get_doc(doc_id)
        if not chunks:
            return "Document not found."
        parts = []
        for c in chunks:
            parts.append(
                f"[chunk {c['chunk_index']}] {c['chunk_text'][:800]}"
            )
        header = f"Document: {chunks[0]['title']}\nURL: {chunks[0]['url']}\nChunks: {len(chunks)}\n"
        return header + "\n\n".join(parts)

    return rag_get_doc


def make_record_signal_tool(evidence_store: EvidenceStore):
    @tool
    def record_signal(
        pain: str,
        workaround: str,
        desired_outcome: str,
        segment: str,
        severity: int,
        willingness_to_pay: int,
        keywords: list[str],
        doc_id: str,
        url: str,
    ) -> str:
        """Record a discovered pain-point signal. Call this for each distinct pain point found.
        severity and willingness_to_pay are 1-5 scales. doc_id is the document hash (before the #)."""
        signal = Signal(
            pain=pain,
            workaround=workaround,
            desired_outcome=desired_outcome,
            segment=segment,
            severity=severity,
            willingness_to_pay=willingness_to_pay,
            keywords=keywords,
            doc_id=doc_id,
            url=url,
        )
        added = evidence_store.add(signal)
        if added:
            return f"Signal recorded: {pain!r} (total: {evidence_store.total_signals})"
        return f"Duplicate doc_id={doc_id!r}, signal not added."

    return record_signal


def make_web_search_tool(token: str):
    from research_loop.search import BraveSearchClient

    client = BraveSearchClient(token=token)

    @tool
    def web_search(query: str, count: int = 10) -> str:
        """Search the live web. Use to find existing products or solutions that address a pain point."""
        try:
            data = client.search(query, count=count)
            results = client.extract_results(data)
        except Exception as exc:
            return f"Search failed: {exc}"
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(
                f"- {r.get('title', '(no title)')}\n"
                f"  URL: {r.get('url', '')}\n"
                f"  {r.get('description', '')}"
            )
        return "\n\n".join(parts)

    return web_search


def make_fetch_page_tool():
    from research_loop.fetch import fetch_page as fetch_url
    from urllib.parse import urlparse

    @tool
    def fetch_page(url: str) -> str:
        """Fetch the full content of a web page. Returns structured output with URL, domain, title, and content.
        Use after web_search to read an existing solution's details, or to fetch pages for indexing."""
        try:
            result = fetch_url(url)
            text = result.get("text", "")[:8000]
            title = result.get("title", "")
            domain = urlparse(url).netloc
            return (
                f"URL: {url}\n"
                f"Domain: {domain}\n"
                f"Title: {title}\n"
                f"Content:\n{text}"
            )
        except Exception as exc:
            return f"Failed to fetch page: {exc}"

    return fetch_page


def make_get_progress_tool(evidence_store: EvidenceStore, max_evidence: int):
    @tool
    def get_progress() -> str:
        """Check current progress: total signals, pain group count, and top themes.
        Use this to decide whether to keep searching or stop."""
        total = evidence_store.total_signals
        groups = evidence_store.pain_groups
        top_themes = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        lines = [
            f"Total signals: {total}/{max_evidence}",
            f"Pain groups: {len(groups)}",
            "Top themes:",
        ]
        for key, signals in top_themes:
            lines.append(f"  - {key!r} ({len(signals)} signals)")
        return "\n".join(lines)

    return get_progress


def make_index_to_rag_tool(conn, table_name: str, namespace: str, embedder, dimension: int, goal: str, indexed_urls: set[str] | None = None):
    """Create a tool that indexes a fetched web page into the RAG database."""
    from index_rag.chunker import chunk_text
    from index_rag.ids import chunk_id as make_chunk_id
    from index_rag.ids import doc_id as make_doc_id
    from index_rag.indexer import PgVectorIndexer, build_vector

    if indexed_urls is None:
        indexed_urls = set()

    indexer = PgVectorIndexer(conn, table_name, dimension)

    @tool
    def index_to_rag(url: str, title: str, text: str, source_domain: str) -> str:
        """Index a fetched web page into the RAG database so it can be searched later.
        Use this after fetch_page to add content to the corpus for analysis.
        Args: url, title, text (page content), source_domain."""
        if not text or not text.strip():
            return "Skipped: empty text content."

        if url in indexed_urls:
            return f"Already indexed: {url}"

        did = make_doc_id(url)
        chunks = chunk_text(text)
        if not chunks:
            return "Skipped: text too short to produce chunks."

        chunk_texts = [c for c in chunks]
        embeddings = embedder.embed(chunk_texts)

        vectors = []
        for i, (ct, emb) in enumerate(zip(chunk_texts, embeddings)):
            cid = make_chunk_id(did, i)
            vectors.append(build_vector(
                cid, emb,
                url=url,
                title=title,
                source_domain=source_domain,
                goal=goal,
                query="",
                chunk_index=i,
                chunk_text=ct,
            ))

        indexer.upsert_batched(vectors, namespace)
        indexed_urls.add(url)
        return f"Indexed {url} → doc_id={did}, {len(chunks)} chunks"

    return index_to_rag
