"""E2E tests: research loop → RAG indexer → PostgreSQL + pgvector."""

from __future__ import annotations

import json
import pathlib
import socket
from unittest.mock import MagicMock

import numpy as np
import pytest

from index_rag.chunker import chunk_text
from index_rag.embedder import DEFAULT_DIMENSION
from index_rag.ids import chunk_id, doc_id
from index_rag.indexer import build_vector, ensure_index, upsert_batched
from index_rag.reader import extract_text, read_jsonl

PG_URL = "postgresql://postgres:postgres@localhost:5433/rag"


# ── Helpers ──────────────────────────────────────────────────────────


def _postgres_reachable(host: str = "localhost", port: int = 5433) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


skip_no_postgres = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="PostgreSQL not running on localhost:5433",
)


class FakeEmbedder:
    """Deterministic embedder that returns unit vectors seeded per chunk index."""

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i, _ in enumerate(texts):
            rng = np.random.default_rng(seed=hash(texts[i]) % (2**31))
            vec = rng.standard_normal(self.dimension)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec.tolist())
        return vectors


def _get_conn():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(PG_URL)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def _drop_table_if_exists(conn, table_name: str) -> None:
    from psycopg import sql

    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {tbl}").format(tbl=sql.Identifier(table_name)))
    conn.commit()


def _build_fake_search(snippets_len: int = 300):
    """Return a mock brave_search function producing results with long snippets."""

    def mock_search(query, count=20, timeout=25, **kwargs):
        return {
            "web": {
                "results": [
                    {
                        "title": f"Result for {query} #{i}",
                        "url": f"https://example.com/{query.replace(' ', '-')}/{i}",
                        "description": "X" * snippets_len,
                        "extra_snippets": [],
                    }
                    for i in range(min(count, 3))
                ]
            }
        }

    return mock_search


def _index_jsonl(jsonl_path, conn, table_name, namespace, embedder=None):
    """Run the index pipeline: read → chunk → embed → upsert. Return vectors list."""
    if embedder is None:
        embedder = FakeEmbedder()

    records = list(read_jsonl(jsonl_path))
    vectors = []
    for rec in records:
        text = extract_text(rec, min_text_len=10)
        if text is None:
            continue
        url = rec.get("url", "")
        did = doc_id(url)
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        embeddings = embedder.embed(chunks)
        for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cid = chunk_id(did, ci)
            v = build_vector(
                cid,
                emb,
                url=url,
                title=rec.get("title", ""),
                source_domain=rec.get("source_domain", ""),
                goal=rec.get("goal", ""),
                query=rec.get("query", ""),
                chunk_index=ci,
                chunk_text=chunk,
            )
            vectors.append(v)

    ensure_index(conn, table_name, DEFAULT_DIMENSION)
    upsert_batched(conn, table_name, vectors, namespace=namespace)
    return vectors


def _query_vectors(conn, table_name, query_vec, namespace, top_k=5):
    """Query vectors by cosine distance, returning matches."""
    from psycopg import sql

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT id, url, title, source_domain, chunk_text, "
                "1 - (embedding <=> %s::vector) AS score "
                "FROM {tbl} WHERE namespace = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s"
            ).format(tbl=sql.Identifier(table_name)),
            (str(query_vec), namespace, str(query_vec), top_k),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "metadata": {"url": r[1], "title": r[2], "source_domain": r[3], "chunk_text": r[4]},
            "score": r[5],
        }
        for r in rows
    ]


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_no_postgres
class TestResearchLoopToPostgresE2E:
    TABLE_NAME = "test_e2e_loop"

    def test_research_loop_to_postgres_e2e(self, monkeypatch, tmp_path):
        """Full pipeline: research loop → JSONL → chunk → embed → upsert → query."""
        from research_loop import loop

        # ── 1. Run research loop with mocked LLM + search ──
        monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "tok")
        monkeypatch.setenv("GROQ_API_KEY", "key")

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [
                {"query": "site:example.com topic-a", "why": "r", "domain": "example.com"},
                {"query": "site:example.com topic-b", "why": "r", "domain": "example.com"},
            ]
        )
        mock_llm.invoke.return_value = mock_response
        monkeypatch.setattr(loop, "build_llm", lambda *a, **kw: mock_llm)
        monkeypatch.setattr(loop, "brave_search", _build_fake_search())

        out = tmp_path / "corpus.jsonl"
        loop.run_loop(
            goal="test e2e",
            domains=["example.com"],
            iterations=1,
            per_iter_queries=2,
            results_per_query=2,
            max_docs=50,
            out=str(out),
            sleep_sec=0,
        )

        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 2, "research loop should produce records"

        # ── 2. Index into real PostgreSQL ──
        conn = _get_conn()
        _drop_table_if_exists(conn, self.TABLE_NAME)

        embedder = FakeEmbedder()
        namespace = "e2e"
        vectors = _index_jsonl(out, conn, self.TABLE_NAME, namespace, embedder)
        assert len(vectors) > 0

        # ── 3. Query and verify ──
        query_vec = vectors[0]["values"]
        matches = _query_vectors(conn, self.TABLE_NAME, query_vec, namespace, top_k=5)
        assert len(matches) > 0, "should return at least one match"

        top = matches[0]
        assert top["score"] > 0.9, f"self-query score should be high, got {top['score']}"
        assert "url" in top["metadata"]
        assert "title" in top["metadata"]
        assert "chunk_text" in top["metadata"]

        # ── 4. Cleanup ──
        _drop_table_if_exists(conn, self.TABLE_NAME)
        conn.close()


@pytest.mark.integration
@skip_no_postgres
class TestIdempotentUpsert:
    TABLE_NAME = "test_e2e_idempotent"

    def test_idempotent_upsert(self, tmp_path):
        """Re-indexing the same JSONL yields the same vector count (no duplicates)."""
        f = tmp_path / "data.jsonl"
        records = [
            {
                "url": f"https://example.com/{i}",
                "title": f"Title {i}",
                "snippet": "W" * 300,
                "source_domain": "example.com",
                "goal": "test",
                "query": "q",
                "dedupe": "unique",
            }
            for i in range(3)
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        conn = _get_conn()
        _drop_table_if_exists(conn, self.TABLE_NAME)

        embedder = FakeEmbedder()
        ns = "idempotent"

        # First upsert
        vecs1 = _index_jsonl(f, conn, self.TABLE_NAME, ns, embedder)

        from psycopg import sql

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {tbl} WHERE namespace = %s").format(
                    tbl=sql.Identifier(self.TABLE_NAME)
                ),
                (ns,),
            )
            count1 = cur.fetchone()[0]

        # Second upsert (same data)
        vecs2 = _index_jsonl(f, conn, self.TABLE_NAME, ns, embedder)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {tbl} WHERE namespace = %s").format(
                    tbl=sql.Identifier(self.TABLE_NAME)
                ),
                (ns,),
            )
            count2 = cur.fetchone()[0]

        assert count1 == count2, f"count changed: {count1} → {count2}"
        assert len(vecs1) == len(vecs2)

        _drop_table_if_exists(conn, self.TABLE_NAME)
        conn.close()


@pytest.mark.integration
@skip_no_postgres
class TestNamespaceIsolation:
    TABLE_NAME = "test_e2e_ns_isolation"

    def test_namespace_isolation(self, tmp_path):
        """Vectors in different namespaces don't bleed into each other's queries."""
        conn = _get_conn()
        _drop_table_if_exists(conn, self.TABLE_NAME)

        embedder = FakeEmbedder()

        # Namespace A
        fa = tmp_path / "a.jsonl"
        fa.write_text(
            json.dumps(
                {
                    "url": "https://a.com/1",
                    "title": "Doc A",
                    "snippet": "A" * 300,
                    "source_domain": "a.com",
                    "goal": "test",
                    "query": "qa",
                    "dedupe": "unique",
                }
            )
            + "\n"
        )
        vecs_a = _index_jsonl(fa, conn, self.TABLE_NAME, "ns-a", embedder)

        # Namespace B
        fb = tmp_path / "b.jsonl"
        fb.write_text(
            json.dumps(
                {
                    "url": "https://b.com/1",
                    "title": "Doc B",
                    "snippet": "B" * 300,
                    "source_domain": "b.com",
                    "goal": "test",
                    "query": "qb",
                    "dedupe": "unique",
                }
            )
            + "\n"
        )
        vecs_b = _index_jsonl(fb, conn, self.TABLE_NAME, "ns-b", embedder)

        # Query ns-a with vec from A → should only get A docs
        matches_a = _query_vectors(conn, self.TABLE_NAME, vecs_a[0]["values"], "ns-a", top_k=10)
        for m in matches_a:
            assert m["metadata"]["source_domain"] == "a.com", (
                f"ns-a returned doc from {m['metadata']['source_domain']}"
            )

        # Query ns-b with vec from B → should only get B docs
        matches_b = _query_vectors(conn, self.TABLE_NAME, vecs_b[0]["values"], "ns-b", top_k=10)
        for m in matches_b:
            assert m["metadata"]["source_domain"] == "b.com", (
                f"ns-b returned doc from {m['metadata']['source_domain']}"
            )

        _drop_table_if_exists(conn, self.TABLE_NAME)
        conn.close()


FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.integration
@skip_no_postgres
class TestIndexerRealEmbeddingsE2E:
    TABLE_NAME = "test_e2e_real_embed"

    def test_index_and_query_with_real_embeddings(self):
        """Full pipeline with real sentence-transformer embeddings + pgvector semantic search."""
        from index_rag.embedder import Embedder

        fixture = FIXTURE_DIR / "inventory-sample.jsonl"
        assert fixture.exists(), f"fixture not found: {fixture}"

        embedder = Embedder()
        conn = _get_conn()
        _drop_table_if_exists(conn, self.TABLE_NAME)

        try:
            # ── 1. Index fixture with real embeddings ──
            vectors = _index_jsonl(fixture, conn, self.TABLE_NAME, "real", embedder)
            assert len(vectors) > 0, "should produce at least one vector"

            # ── 2. Verify row count ──
            from psycopg import sql

            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {tbl} WHERE namespace = %s").format(
                        tbl=sql.Identifier(self.TABLE_NAME)
                    ),
                    ("real",),
                )
                row_count = cur.fetchone()[0]
            assert row_count == len(vectors), f"expected {len(vectors)} rows, got {row_count}"

            # ── 3. Semantic query ──
            query_text = "inventory management feature requests"
            query_vec = embedder.embed([query_text])[0]
            matches = _query_vectors(conn, self.TABLE_NAME, query_vec, "real", top_k=5)

            assert len(matches) > 0, "semantic query should return results"

            # ── 4. Assert metadata fields present ──
            for m in matches:
                assert m["score"] > 0, f"score should be positive, got {m['score']}"
                md = m["metadata"]
                assert md["url"], "url should be non-empty"
                assert md["title"], "title should be non-empty"
                assert md["source_domain"], "source_domain should be non-empty"
                assert md["chunk_text"], "chunk_text should be non-empty"

            # ── 5. Assert ordering by score descending ──
            scores = [m["score"] for m in matches]
            assert scores == sorted(scores, reverse=True), "results should be ordered by score desc"

        finally:
            _drop_table_if_exists(conn, self.TABLE_NAME)
            conn.close()
