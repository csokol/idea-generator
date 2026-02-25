"""End-to-end tests for the agent graph with mocked LLM and search API."""

from __future__ import annotations

from typing import Any, Sequence

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent, ToolNode

from agent.tools import brave_search


class FakeChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel with a no-op bind_tools for create_react_agent."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        return self

FAKE_SEARCH_RESPONSE = {
    "grounding": {
        "generic": [
            {
                "title": "Python 3.14 Released",
                "url": "https://python.org/news",
                "snippets": ["Python 3.14 was released today with exciting new features."],
            },
        ]
    }
}


class FakeHttpResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or FAKE_SEARCH_RESPONSE
        self.text = '{"error": "server error"}'

    def json(self):
        return self._data


def _build_test_agent(llm_responses):
    fake_llm = FakeChatModel(responses=llm_responses)
    return create_react_agent(fake_llm, [brave_search], prompt="You are a helpful assistant.")


def test_agent_calls_tool_and_responds(monkeypatch):
    """LLM decides to search, gets results, then produces a final answer."""
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeHttpResponse())

    agent = _build_test_agent([
        AIMessage(
            content="Let me search for that.",
            tool_calls=[
                {"name": "brave_search", "args": {"query": "python 3.14 release"}, "id": "call_1"},
            ],
        ),
        AIMessage(content="Python 3.14 was released with exciting new features."),
    ])

    result = agent.invoke({"messages": [("user", "What's new in Python 3.14?")]})
    messages = result["messages"]

    # user -> ai (tool call) -> tool result -> ai (final)
    assert len(messages) == 4
    assert messages[0].content == "What's new in Python 3.14?"
    assert messages[1].tool_calls[0]["name"] == "brave_search"
    assert "Python 3.14" in messages[2].content  # tool result
    assert "Python 3.14" in messages[-1].content  # final answer


def test_agent_answers_without_tool(monkeypatch):
    """LLM answers directly without calling any tool."""
    agent = _build_test_agent([
        AIMessage(content="The capital of France is Paris."),
    ])

    result = agent.invoke({"messages": [("user", "What is the capital of France?")]})
    messages = result["messages"]

    assert len(messages) == 2
    assert "Paris" in messages[-1].content


def test_agent_handles_tool_error(monkeypatch):
    """When the search API fails, the error is fed back to the LLM which recovers."""
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeHttpResponse(status_code=500))

    fake_llm = FakeChatModel(responses=[
        AIMessage(
            content="Let me search.",
            tool_calls=[
                {"name": "brave_search", "args": {"query": "test"}, "id": "call_1"},
            ],
        ),
        AIMessage(content="Sorry, I couldn't complete the search. Please try again later."),
    ])
    tool_node = ToolNode([brave_search], handle_tool_errors=True)
    agent = create_react_agent(fake_llm, tool_node, prompt="You are a helpful assistant.")

    result = agent.invoke({"messages": [("user", "Search for test")]})
    messages = result["messages"]

    assert len(messages) == 4
    assert "error" in messages[2].content.lower()
    assert "Sorry" in messages[-1].content


def test_agent_multi_turn_conversation(monkeypatch):
    """Agent handles a multi-turn conversation maintaining history."""
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeHttpResponse())

    fake_llm = FakeChatModel(responses=[
        AIMessage(
            content="Searching for news.",
            tool_calls=[
                {"name": "brave_search", "args": {"query": "today's news"}, "id": "call_1"},
            ],
        ),
        AIMessage(content="Python 3.14 was released today."),
        AIMessage(content="It includes new features like improved performance and type system enhancements."),
    ])

    agent = create_react_agent(fake_llm, [brave_search], prompt="You are a helpful assistant.")

    # Turn 1
    result1 = agent.invoke({"messages": [("user", "What happened today?")]})
    messages = result1["messages"]
    assert "Python 3.14" in messages[-1].content

    # Turn 2: continue with history
    messages.append(("user", "Tell me more about it."))
    result2 = agent.invoke({"messages": messages})
    messages2 = result2["messages"]
    assert "features" in messages2[-1].content.lower()


def test_agent_multiple_tool_calls(monkeypatch):
    """LLM makes two sequential searches before answering."""
    monkeypatch.setenv("BRAVE_SEARCH_TOKEN", "fake-token")

    call_count = 0
    responses = [
        FakeHttpResponse(data={
            "grounding": {"generic": [
                {"title": "Python News", "url": "https://python.org", "snippets": ["Python update"]},
            ]}
        }),
        FakeHttpResponse(data={
            "grounding": {"generic": [
                {"title": "Rust News", "url": "https://rust-lang.org", "snippets": ["Rust update"]},
            ]}
        }),
    ]

    def fake_get(*a, **kw):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    monkeypatch.setattr(httpx, "get", fake_get)

    agent = _build_test_agent([
        AIMessage(
            content="Let me search for both.",
            tool_calls=[
                {"name": "brave_search", "args": {"query": "python news"}, "id": "call_1"},
                {"name": "brave_search", "args": {"query": "rust news"}, "id": "call_2"},
            ],
        ),
        AIMessage(content="Here are updates on both Python and Rust."),
    ])

    result = agent.invoke({"messages": [("user", "News about Python and Rust")]})
    messages = result["messages"]

    assert call_count == 2
    assert "Python" in messages[-1].content
    assert "Rust" in messages[-1].content
