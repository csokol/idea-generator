"""PostgreSQL + pgvector upsert orchestration (table creation, batched upsert)."""

from __future__ import annotations

from psycopg import sql

METADATA_MAX_BYTES = 40_000


class PgVectorIndexer:
    def __init__(self, conn, table_name: str, dimension: int):
        self.conn = conn
        self.table_name = table_name
        self.dimension = dimension

    def ensure_index(self) -> None:
        """Create the pgvector extension, table, and indexes if they don't exist.

        If the table already exists with a different embedding dimension,
        it is dropped and recreated with the requested dimension.
        """
        table = sql.Identifier(self.table_name)
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))

            # Check if table exists with a different vector dimension
            cur.execute(
                "SELECT a.atttypmod FROM pg_attribute a "
                "JOIN pg_class c ON a.attrelid = c.oid "
                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                "WHERE n.nspname = 'public' AND c.relname = %s AND a.attname = 'embedding'",
                (self.table_name,),
            )
            row = cur.fetchone()
            if row is not None and row[0] != self.dimension:
                print(
                    f"  Table '{self.table_name}' has dimension {row[0]}, "
                    f"need {self.dimension} — recreating."
                )
                cur.execute(sql.SQL("DROP TABLE {tbl}").format(tbl=table))

            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {tbl} ("
                    "  id            TEXT        NOT NULL,"
                    "  namespace     TEXT        NOT NULL DEFAULT '',"
                    "  embedding     vector({dim}) NOT NULL,"
                    "  url           TEXT        NOT NULL DEFAULT '',"
                    "  title         TEXT        NOT NULL DEFAULT '',"
                    "  source_domain TEXT        NOT NULL DEFAULT '',"
                    "  goal          TEXT        NOT NULL DEFAULT '',"
                    "  query         TEXT        NOT NULL DEFAULT '',"
                    "  chunk_index   INTEGER     NOT NULL DEFAULT 0,"
                    "  chunk_text    TEXT        NOT NULL DEFAULT '',"
                    "  PRIMARY KEY (id, namespace)"
                    ")"
                ).format(tbl=table, dim=sql.SQL(str(self.dimension)))
            )
            # Index for namespace filtering
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} (namespace)"
                ).format(
                    idx=sql.Identifier(f"{self.table_name}_namespace_idx"),
                    tbl=table,
                )
            )
        self.conn.commit()

    def upsert_batched(self, vectors: list[dict], namespace: str, batch_size: int = 100) -> int:
        """Upsert vectors in batches. Return total upserted count."""
        table = sql.Identifier(self.table_name)
        query = sql.SQL(
            "INSERT INTO {tbl} (id, namespace, embedding, url, title, source_domain, goal, query, chunk_index, chunk_text) "
            "VALUES (%(id)s, %(namespace)s, %(embedding)s, %(url)s, %(title)s, %(source_domain)s, %(goal)s, %(query)s, %(chunk_index)s, %(chunk_text)s) "
            "ON CONFLICT (id, namespace) DO UPDATE SET "
            "  embedding     = EXCLUDED.embedding,"
            "  url           = EXCLUDED.url,"
            "  title         = EXCLUDED.title,"
            "  source_domain = EXCLUDED.source_domain,"
            "  goal          = EXCLUDED.goal,"
            "  query         = EXCLUDED.query,"
            "  chunk_index   = EXCLUDED.chunk_index,"
            "  chunk_text    = EXCLUDED.chunk_text"
        ).format(tbl=table)

        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            params = []
            for v in batch:
                m = v["metadata"]
                params.append({
                    "id": v["id"],
                    "namespace": namespace,
                    "embedding": str(v["values"]),
                    "url": m.get("url", ""),
                    "title": m.get("title", ""),
                    "source_domain": m.get("source_domain", ""),
                    "goal": m.get("goal", ""),
                    "query": m.get("query", ""),
                    "chunk_index": m.get("chunk_index", 0),
                    "chunk_text": m.get("chunk_text", ""),
                })
            with self.conn.cursor() as cur:
                cur.executemany(query, params)
            self.conn.commit()
            total += len(batch)
        return total


# Backward-compatible shims
def ensure_index(conn, table_name: str, dimension: int) -> None:
    """Create the pgvector extension, table, and indexes if they don't exist."""
    PgVectorIndexer(conn, table_name, dimension).ensure_index()


def _truncate(value: str, max_bytes: int = METADATA_MAX_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def build_vector(
    vector_id: str,
    values: list[float],
    *,
    url: str,
    title: str,
    source_domain: str,
    goal: str,
    query: str,
    chunk_index: int,
    chunk_text: str,
) -> dict:
    """Build a single vector dict with metadata."""
    return {
        "id": vector_id,
        "values": values,
        "metadata": {
            "url": url,
            "title": _truncate(title, 1000),
            "source_domain": source_domain,
            "goal": goal,
            "query": _truncate(query, 1000),
            "chunk_index": chunk_index,
            "chunk_text": _truncate(chunk_text),
        },
    }


def upsert_batched(
    conn, table_name: str, vectors: list[dict], namespace: str, batch_size: int = 100
) -> int:
    """Upsert vectors in batches. Return total upserted count."""
    return PgVectorIndexer(conn, table_name, 0).upsert_batched(vectors, namespace, batch_size)
