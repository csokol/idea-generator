"""JSONL record building, dedup, and file writing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, host, path, parsed.params, parsed.query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Deduplicator:
    def __init__(self) -> None:
        self.seen_url_hashes: set[str] = set()

    def is_duplicate(self, url: str) -> bool:
        return url_hash(url) in self.seen_url_hashes

    def mark_seen(self, url: str) -> None:
        self.seen_url_hashes.add(url_hash(url))


def build_record(
    *,
    run_id: str,
    goal: str,
    domains: list[str],
    iteration: int,
    query: str,
    rank: int,
    url: str,
    title: str,
    snippet: str,
    source_domain: str,
    content: str | None = None,
    dedupe: str = "unique",
    errors: list[str] | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "goal": goal,
        "domains": domains,
        "iteration": iteration,
        "query": query,
        "rank": rank,
        "url": url,
        "title": title,
        "snippet": snippet,
        "source_domain": source_domain,
        "content": content,
        "dedupe": dedupe,
        "errors": errors or [],
    }


def append_jsonl(record: dict, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
