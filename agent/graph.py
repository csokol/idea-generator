"""LangGraph ReAct agent with pluggable LLM and Brave Search tool."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from agent.tools import brave_search
from llm_factory import build_llm

DEFAULT_SYSTEM = "You are a helpful assistant. Use the brave_search tool to look up current information when needed."


def build_agent(
    model_name: str | None = None,
    system_message: str = DEFAULT_SYSTEM,
    provider: str = "gemini",
):
    llm = build_llm(provider=provider, model=model_name)
    return create_react_agent(llm, [brave_search], prompt=system_message)


def run_agent(query: str, **kwargs) -> str:
    agent = build_agent(**kwargs)
    result = agent.invoke({"messages": [("user", query)]})
    return result["messages"][-1].content
