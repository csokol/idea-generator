"""RAG store query layer — semantic search over pgvector chunks."""

from __future__ import annotations

import logging

from psycopg import sql

from typing import Protocol


class _Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

log = logging.getLogger(__name__)


class RagStore:
    def __init__(
        self,
        conn,
        table_name: str,
        namespace: str,
        embedder: _Embedder,
    ) -> None:
        self.conn = conn
        self.table_name = table_name
        self.namespace = namespace
        self.embedder = embedder
        self._embed_cache: dict[str, list[float]] = {}
        self._search_cache: dict[tuple, list[dict]] = {}

    def _get_embedding(self, text: str) -> list[float]:
        if text not in self._embed_cache:
            self._embed_cache[text] = self.embedder.embed([text])[0]
        return self._embed_cache[text]

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        domains: list[str] | None = None,
    ) -> list[dict]:
        cache_key = (query_text, top_k, tuple(sorted(domains)) if domains else ())
        if cache_key in self._search_cache:
            log.debug("RAG search cache hit: %r", query_text)
            return self._search_cache[cache_key]

        log.debug("RAG search: query=%r top_k=%d domains=%s", query_text, top_k, domains)
        embedding = self._get_embedding(query_text)
        vec_literal = str(embedding)
        tbl = sql.Identifier(self.table_name)

        conditions = [sql.SQL("namespace = {ns}").format(ns=sql.Literal(self.namespace))]
        if domains:
            conditions.append(
                sql.SQL("source_domain IN ({doms})").format(
                    doms=sql.SQL(", ").join(sql.Literal(d) for d in domains)
                )
            )

        where = sql.SQL(" AND ").join(conditions)

        query = sql.SQL(
            "SELECT id, url, title, source_domain, chunk_index, chunk_text, "
            "1 - (embedding <=> {vec}::vector) AS score "
            "FROM {tbl} WHERE {where} "
            "ORDER BY embedding <=> {vec}::vector LIMIT {limit}"
        ).format(
            tbl=tbl,
            vec=sql.Literal(vec_literal),
            where=where,
            limit=sql.Literal(top_k),
        )

        with self.conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        hits = [dict(zip(columns, row)) for row in rows]
        self._search_cache[cache_key] = hits
        log.info("RAG search returned %d hits (top score=%.3f)", len(hits), hits[0]["score"] if hits else 0)
        for h in hits:
            log.debug("  [%s] score=%.3f %s", h["id"], h["score"], h["url"])
        return hits

    def get_doc(self, doc_id: str) -> list[dict]:
        """Get all chunks for a document by its doc_id prefix."""
        tbl = sql.Identifier(self.table_name)
        query = sql.SQL(
            "SELECT id, url, title, source_domain, chunk_index, chunk_text "
            "FROM {tbl} WHERE namespace = {ns} AND id LIKE {pattern} "
            "ORDER BY chunk_index"
        ).format(
            tbl=tbl,
            ns=sql.Literal(self.namespace),
            pattern=sql.Literal(doc_id + "#%"),
        )

        with self.conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]
