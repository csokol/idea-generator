"""Optional page fetching: HTML -> plain text."""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)

MAX_TEXT_LEN = 50_000
DEFAULT_USER_AGENT = "ResearchLoop/0.1 (compatible; bot)"


def fetch_page(
    url: str,
    timeout: int = 10,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    log.info("[FETCH START] %s", url)
    t0 = time.monotonic()
    try:
        headers = {"User-Agent": user_agent}
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        content_type = resp.headers.get("content-type", "")
        charset = resp.charset_encoding or "utf-8"

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = text[:MAX_TEXT_LEN]

        elapsed = time.monotonic() - t0
        log.info("[FETCH DONE] %s — status=%d, %d chars (%.1fs)", url, resp.status_code, len(text), elapsed)
        return {
            "fetched": True,
            "status_code": resp.status_code,
            "content_type": content_type,
            "charset": charset,
            "text": text,
            "text_len": len(text),
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.warning("[FETCH FAIL] %s — %s (%.1fs)", url, exc, elapsed)
        return {
            "fetched": False,
            "status_code": None,
            "content_type": None,
            "charset": None,
            "text": None,
            "text_len": 0,
            "error": str(exc),
        }
