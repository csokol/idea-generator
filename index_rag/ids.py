"""Stable/canonical doc and chunk ID generation."""

from __future__ import annotations

from research_loop.records import url_hash


def doc_id(url: str) -> str:
    """SHA256 of the canonical URL — deterministic across runs."""
    return url_hash(url)


def chunk_id(doc_id: str, chunk_index: int) -> str:
    """Unique vector ID: ``<doc_hash>#<chunk_index>``."""
    return f"{doc_id}#{chunk_index}"
