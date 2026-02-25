"""CLI entrypoint for the RAG indexer."""

from __future__ import annotations

import argparse
import sys

from index_rag.chunker import chunk_text
from index_rag.embedder import DEFAULT_DIMENSION, DEFAULT_MODEL, GOOGLE_DEFAULT_DIMENSION, OPENAI_DEFAULT_DIMENSION, build_embedder
from index_rag.ids import chunk_id, doc_id
from index_rag.indexer import build_vector, ensure_index, upsert_batched
from index_rag.reader import extract_text, read_jsonl

DEFAULT_PG_URL = "postgresql://postgres:postgres@localhost:5433/rag"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index JSONL research corpus into PostgreSQL + pgvector")
    p.add_argument("--in", dest="input", required=True, help="Path to JSONL file")
    p.add_argument("--index", required=True, help="Table name for vectors")
    p.add_argument("--namespace", default="", help="Namespace (stored as column value)")
    p.add_argument("--pg-url", default=DEFAULT_PG_URL, help="PostgreSQL connection URL")
    p.add_argument("--dimension", type=int, default=OPENAI_DEFAULT_DIMENSION, help="Embedding dimension")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--chunk-overlap", type=int, default=120)
    p.add_argument("--max-chunks-per-doc", type=int, default=50)
    p.add_argument("--min-text-len", type=int, default=200)
    p.add_argument("--model", default=None, help="Embedding model name")
    p.add_argument("--embed-provider", choices=["local", "google", "openai"], default="openai", help="Embedding provider")
    p.add_argument("--dry-run", action="store_true", help="Skip database upserts")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

    # 1. Read and filter JSONL
    docs: list[tuple[dict, str]] = []
    skipped = 0
    for record in read_jsonl(args.input):
        text = extract_text(record, min_text_len=args.min_text_len)
        if text is None:
            skipped += 1
            continue
        docs.append((record, text))

    print(f"Documents: {len(docs)} usable, {skipped} skipped")

    # 2. Chunk
    all_vectors_meta: list[tuple[str, str, dict]] = []  # (vector_id, chunk_text, record)
    total_chunks = 0
    for record, text in docs:
        did = doc_id(record["url"])
        chunks = chunk_text(
            text,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks_per_doc=args.max_chunks_per_doc,
        )
        for i, ct in enumerate(chunks):
            cid = chunk_id(did, i)
            all_vectors_meta.append((cid, ct, record))
        total_chunks += len(chunks)

    print(f"Chunks: {total_chunks} total vectors to index")

    if not all_vectors_meta:
        print("Nothing to index.")
        return

    # 3. Embed — auto-adjust dimension when not explicitly set
    if args.embed_provider == "local" and args.dimension == OPENAI_DEFAULT_DIMENSION:
        args.dimension = DEFAULT_DIMENSION
    elif args.embed_provider == "google" and args.dimension == OPENAI_DEFAULT_DIMENSION:
        args.dimension = GOOGLE_DEFAULT_DIMENSION

    embedder = build_embedder(provider=args.embed_provider, model=args.model, dimension=args.dimension if args.embed_provider in ("google", "openai") else None)
    print(f"Embedding with provider={args.embed_provider!r} model={args.model or 'default'}...")
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

    if args.dry_run:
        print(f"Dry-run complete. Would upsert {len(vectors)} vectors into '{args.index}' namespace='{args.namespace}'")
        return

    # 5. Upsert to PostgreSQL
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(args.pg_url)
    register_vector(conn)
    ensure_index(conn, args.index, args.dimension)
    count = upsert_batched(conn, args.index, vectors, namespace=args.namespace, batch_size=args.batch_size)
    conn.close()
    print(f"Upserted {count} vectors into '{args.index}' namespace='{args.namespace}'")


if __name__ == "__main__":
    main()
