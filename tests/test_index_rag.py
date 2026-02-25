"""Tests for the index_rag package."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from index_rag.chunker import chunk_text
from index_rag.embedder import GoogleEmbedder, LocalEmbedder, OpenAIEmbedder, build_embedder
from index_rag.ids import chunk_id, doc_id
from index_rag.indexer import PgVectorIndexer, build_vector, ensure_index, upsert_batched
from index_rag.reader import extract_text, read_jsonl


# ── Reader ──────────────────────────────────────────────────────────


class TestReadJsonl:
    def test_reads_valid_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"url": "https://a.com", "title": "A"}) + "\n"
            + json.dumps({"url": "https://b.com", "title": "B"}) + "\n"
        )
        records = list(read_jsonl(f))
        assert len(records) == 2
        assert records[0]["url"] == "https://a.com"

    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
        records = list(read_jsonl(f))
        assert len(records) == 2

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "blank.jsonl"
        f.write_text('{"a":1}\n\n\n{"b":2}\n')
        records = list(read_jsonl(f))
        assert len(records) == 2


class TestExtractText:
    def test_concatenates_fields(self):
        record = {"title": "T", "snippet": "S" * 100, "content": "C" * 100, "dedupe": "unique"}
        text = extract_text(record, min_text_len=10)
        assert text is not None
        assert "T" in text
        assert "S" in text
        assert "C" in text

    def test_filters_duplicates(self):
        record = {"title": "T", "snippet": "S" * 200, "dedupe": "duplicate"}
        assert extract_text(record) is None

    def test_filters_short_text(self):
        record = {"title": "Hi", "snippet": "short", "dedupe": "unique"}
        assert extract_text(record, min_text_len=200) is None

    def test_missing_fields(self):
        record = {"title": "T" * 300, "dedupe": "unique"}
        text = extract_text(record, min_text_len=10)
        assert text is not None


# ── Chunker ─────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=900)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_respects_chunk_size(self):
        text = "word " * 500  # 2500 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
        for c in chunks:
            assert len(c) <= 200

    def test_overlap(self):
        text = "A " * 500
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=50)
        assert len(chunks) > 1
        # With overlap, more chunks than without
        no_overlap = chunk_text(text, chunk_size=100, chunk_overlap=0)
        assert len(chunks) > len(no_overlap)

    def test_max_chunks_per_doc(self):
        text = "word " * 10000
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0, max_chunks_per_doc=5)
        assert len(chunks) == 5

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_exact_chunk_size(self):
        text = "a" * 900
        chunks = chunk_text(text, chunk_size=900)
        assert len(chunks) == 1


# ── IDs ─────────────────────────────────────────────────────────────


class TestIds:
    def test_doc_id_deterministic(self):
        id1 = doc_id("https://example.com/page")
        id2 = doc_id("https://example.com/page")
        assert id1 == id2
        assert len(id1) == 64

    def test_doc_id_canonical(self):
        id1 = doc_id("https://Example.COM/page/#frag")
        id2 = doc_id("https://example.com/page")
        assert id1 == id2

    def test_different_urls_different_ids(self):
        assert doc_id("https://a.com/1") != doc_id("https://a.com/2")

    def test_chunk_id_format(self):
        cid = chunk_id("abc123", 5)
        assert cid == "abc123#5"

    def test_different_chunks_different_ids(self):
        assert chunk_id("abc", 0) != chunk_id("abc", 1)


# ── Indexer ─────────────────────────────────────────────────────────


class TestBuildVector:
    def test_has_required_fields(self):
        v = build_vector(
            "id1", [0.1, 0.2],
            url="https://a.com", title="T", source_domain="a.com",
            goal="g", query="q", chunk_index=0, chunk_text="text",
        )
        assert v["id"] == "id1"
        assert v["values"] == [0.1, 0.2]
        assert v["metadata"]["url"] == "https://a.com"
        assert v["metadata"]["chunk_text"] == "text"


class TestEnsureIndex:
    def test_creates_extension_table_and_index(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # table does not exist yet
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        indexer = PgVectorIndexer(mock_conn, "my_table", 384)
        indexer.ensure_index()

        # Should execute 4 statements: CREATE EXTENSION, dimension check, CREATE TABLE, CREATE INDEX
        assert mock_cursor.execute.call_count == 4
        mock_conn.commit.assert_called_once()

        # Check SQL contains expected fragments
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        sql_text = " ".join(calls)
        assert "vector" in sql_text.lower()

    def test_shim_delegates(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        ensure_index(mock_conn, "my_table", 384)
        assert mock_cursor.execute.call_count == 4


class TestUpsertBatched:
    def test_batching(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        vectors = [
            {"id": str(i), "values": [0.0], "metadata": {"url": "", "title": "", "source_domain": "", "goal": "", "query": "", "chunk_index": 0, "chunk_text": ""}}
            for i in range(250)
        ]
        indexer = PgVectorIndexer(mock_conn, "test_table", 384)
        count = indexer.upsert_batched(vectors, namespace="ns", batch_size=100)
        assert count == 250
        # 3 batches: 100 + 100 + 50
        assert mock_cursor.executemany.call_count == 3
        assert mock_conn.commit.call_count == 3

    def test_shim_delegates(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        vectors = [
            {"id": "0", "values": [0.0], "metadata": {"url": "", "title": "", "source_domain": "", "goal": "", "query": "", "chunk_index": 0, "chunk_text": ""}}
        ]
        count = upsert_batched(mock_conn, "test_table", vectors, namespace="ns")
        assert count == 1


# ── Dry-run pipeline ───────────────────────────────────────────────


class TestDryRunPipeline:
    def test_full_dry_run(self, tmp_path, capsys):
        """Full pipeline in dry-run mode: read, chunk, embed (mocked), no DB."""
        # Create test JSONL
        f = tmp_path / "test.jsonl"
        records = [
            {"url": "https://a.com/1", "title": "Title One", "snippet": "S" * 300,
             "source_domain": "a.com", "goal": "test", "query": "q1", "dedupe": "unique"},
            {"url": "https://a.com/2", "title": "Title Two", "snippet": "S" * 300,
             "source_domain": "a.com", "goal": "test", "query": "q2", "dedupe": "unique"},
            {"url": "https://a.com/3", "title": "Dup", "snippet": "S" * 300,
             "source_domain": "a.com", "goal": "test", "query": "q3", "dedupe": "duplicate"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        # Mock SentenceTransformer
        mock_st = MagicMock()
        mock_st.return_value.encode.return_value = __import__("numpy").array(
            [[0.1] * 384] * 100  # enough for any number of chunks
        )

        with patch("sentence_transformers.SentenceTransformer", mock_st):
            from index_rag.cli import main
            main(["--in", str(f), "--index", "test", "--namespace", "dev", "--embed-provider", "local", "--dry-run"])

        output = capsys.readouterr().out
        assert "2 usable" in output
        assert "1 skipped" in output
        assert "Dry-run complete" in output


# ── Embedder factory ──────────────────────────────────────────────


class TestBuildEmbedder:
    def test_local_returns_local_embedder(self):
        mock_st = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", mock_st):
            emb = build_embedder("local")
        assert isinstance(emb, LocalEmbedder)

    def test_google_returns_google_embedder(self):
        mock_cls = MagicMock()
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", mock_cls):
            emb = build_embedder("google")
        assert isinstance(emb, GoogleEmbedder)
        mock_cls.assert_called_once_with(model="gemini-embedding-001")

    def test_google_passes_dimension(self):
        mock_cls = MagicMock()
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", mock_cls):
            emb = build_embedder("google", dimension=384)
        mock_cls.assert_called_once_with(model="gemini-embedding-001", output_dimensionality=384)

    def test_google_custom_model(self):
        mock_cls = MagicMock()
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", mock_cls):
            build_embedder("google", model="models/custom-embed")
        mock_cls.assert_called_once_with(model="models/custom-embed")

    def test_openai_returns_openai_embedder(self):
        mock_cls = MagicMock()
        with patch("langchain_openai.OpenAIEmbeddings", mock_cls):
            emb = build_embedder("openai")
        assert isinstance(emb, OpenAIEmbedder)
        mock_cls.assert_called_once_with(model="text-embedding-3-small")

    def test_openai_passes_dimension(self):
        mock_cls = MagicMock()
        with patch("langchain_openai.OpenAIEmbeddings", mock_cls):
            emb = build_embedder("openai", dimension=512)
        mock_cls.assert_called_once_with(model="text-embedding-3-small", dimensions=512)

    def test_openai_custom_model(self):
        mock_cls = MagicMock()
        with patch("langchain_openai.OpenAIEmbeddings", mock_cls):
            build_embedder("openai", model="text-embedding-3-large")
        mock_cls.assert_called_once_with(model="text-embedding-3-large")

    def test_openai_embed_delegates_to_embed_documents(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_cls.return_value = mock_instance
        with patch("langchain_openai.OpenAIEmbeddings", mock_cls):
            emb = build_embedder("openai")
        result = emb.embed(["hello", "world"])
        mock_instance.embed_documents.assert_called_once_with(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embed provider"):
            build_embedder("unknown_provider")

    def test_google_embed_delegates_to_embed_documents(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_cls.return_value = mock_instance
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", mock_cls):
            emb = build_embedder("google")
        result = emb.embed(["hello", "world"])
        mock_instance.embed_documents.assert_called_once_with(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
