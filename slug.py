"""Generate a short, human-friendly slug from a research goal using an LLM."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

MAX_SLUG_LEN = 10
MAX_MODEL_TAG_LEN = 20

_SLUG_PROMPT = (
    "Generate a single filesystem-safe slug (max 10 chars, lowercase, "
    "alphanumeric and hyphens only, no leading/trailing hyphens) that "
    "summarises this research goal. Reply with ONLY the slug, nothing else.\n\n"
    "Goal: {goal}"
)


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return str(content)


def _sanitize(raw: str) -> str:
    """Lowercase, strip non-alphanumeric/hyphen chars, truncate."""
    slug = re.sub(r"[^a-z0-9-]", "", raw.strip().lower())
    slug = slug.strip("-")
    return slug[:MAX_SLUG_LEN]


def _fallback_slug(goal: str) -> str:
    """Derive a slug from the goal without an LLM."""
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    return slug[:MAX_SLUG_LEN]


def generate_slug(goal: str, llm: BaseChatModel) -> str:
    """Ask the LLM for a short slug; fall back to sanitized goal on failure."""
    try:
        resp = llm.invoke(_SLUG_PROMPT.format(goal=goal))
        slug = _sanitize(_content_to_str(resp.content) if hasattr(resp, "content") else str(resp))
        if slug:
            return slug
    except Exception:
        pass
    return _fallback_slug(goal)


def model_tag(provider: str, model: str | None = None) -> str:
    """Return a short, filesystem-safe tag like 'groq-qwen3-32b'."""
    from llm_factory import PROVIDERS

    name = model or PROVIDERS.get(provider, {}).get("default_model", "unknown")
    # Strip common prefixes/vendor paths (e.g. "qwen/qwen3-32b" → "qwen3-32b")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    tag = re.sub(r"[^a-z0-9]+", "-", f"{provider}-{name}".lower()).strip("-")
    return tag[:MAX_MODEL_TAG_LEN]


def generate_slug_openai(goal: str, model: str = "gpt-4.1-nano") -> str:
    """Generate a slug using the OpenAI Python SDK directly (no LangChain)."""
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _SLUG_PROMPT.format(goal=goal)}],
            max_tokens=20,
            temperature=0.3,
        )
        slug = _sanitize(resp.choices[0].message.content or "")
        if resp.usage:
            from token_tracker import get_tracker
            get_tracker().record("openai", model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        if slug:
            return slug
    except Exception:
        pass
    return _fallback_slug(goal)
