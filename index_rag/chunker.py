"""Text chunking with overlap, splitting on whitespace boundaries."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    max_chunks_per_doc: int = 50,
) -> list[str]:
    """Split *text* into overlapping chunks on whitespace boundaries."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length and len(chunks) < max_chunks_per_doc:
        end = min(start + chunk_size, length)

        # Try to break on whitespace (look backward from end)
        if end < length:
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws

        chunks.append(text[start:end].strip())

        # If we reached the end of the text, stop
        if end >= length:
            break

        # Advance by (chunk_size - overlap), but at least 1 char
        step = max(end - start - chunk_overlap, 1)
        start += step

    return [c for c in chunks if c]
