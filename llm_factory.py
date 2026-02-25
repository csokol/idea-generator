"""Shared LLM factory for pluggable providers."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

PROVIDERS = {
    "groq": {
        "env_var": "GROQ_API_KEY",
        "default_model": "qwen/qwen3-32b",
    },
    "gemini": {
        "env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-3-flash-preview",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4.1",
    },
}


def build_llm(provider: str = "gemini", model: str | None = None) -> BaseChatModel:
    """Build a LangChain chat model for the given provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")

    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_var"], "")
    if not api_key:
        raise RuntimeError(f"{cfg['env_var']} environment variable is not set.")

    model_name = model or cfg["default_model"]

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model_name, api_key=api_key)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider '{provider}'")

    from token_tracker import TokenTrackingHandler, get_tracker

    llm.callbacks = [TokenTrackingHandler(get_tracker(), provider, model_name)]
    return llm
