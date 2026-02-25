---
name: langgraph
description: Reference documentation for LangGraph v1.0+ agent APIs, migration from deprecated create_react_agent, and best practices. Load this skill when working with LangGraph agents, tools, or the insights/agent modules.
user-invocable: true
disable-model-invocation: false
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# LangGraph v1.0+ Agent API Reference

## Migration: `create_react_agent` → `create_agent`

`create_react_agent` from `langgraph.prebuilt` is **deprecated** since LangGraph v1.0 (late 2025). It still works but will be removed in v2.0.

### Old (deprecated)

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools, prompt="You are a helpful assistant.")
result = agent.invoke({"messages": [HumanMessage(content="hello")]})
```

### New (LangGraph v1.0+)

```python
from langchain.agents import create_agent

agent = create_agent(model, tools, system_prompt="You are a helpful assistant.")
result = agent.invoke({"messages": [HumanMessage(content="hello")]})
```

**Requires**: `langchain>=1.0` and `langgraph>=1.0`

---

## `create_agent` API Signature

```python
from langchain.agents import create_agent

agent = create_agent(
    model,                    # str like "openai:gpt-4o" OR BaseChatModel instance
    tools,                    # list of tools (@tool decorated functions or Tool objects)
    system_prompt=None,       # str (replaces old `prompt=` parameter)
    name=None,                # str — agent identifier (useful in multi-agent systems)
    response_format=None,     # Pydantic model for structured final output
    state_schema=None,        # TypedDict for custom state (Pydantic no longer supported)
    middleware=None,           # list of middleware objects
)
```

Returns a `CompiledStateGraph` (same as before — supports `.invoke()`, `.stream()`, `.astream()`).

---

## Key Differences from `create_react_agent`

| Aspect | Old (`create_react_agent`) | New (`create_agent`) |
|--------|---------------------------|---------------------|
| **Import** | `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` |
| **Prompt param** | `prompt=` (str or SystemMessage) | `system_prompt=` (str only) |
| **Model param** | `BaseChatModel` object only | String `"provider:model"` OR `BaseChatModel` |
| **Streaming node** | `"agent"` | `"model"` |
| **State schema** | Pydantic or TypedDict | TypedDict only |
| **Dynamic prompts** | Function arg to `prompt=` | `@dynamic_prompt` middleware |
| **Tool error handling** | Handler functions | `@wrap_tool_call` middleware |
| **Pre/post hooks** | Not available | `@before_model` / `@after_model` middleware |
| **Structured output** | Not built-in | `response_format=` parameter |

---

## Middleware System (New in v1.0)

### Decorator-based

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, before_model, after_model, wrap_tool_call

@dynamic_prompt
def build_prompt(req):
    return f"You are helping with: {req.state.get('context', 'general tasks')}"

@before_model
def log_request(req):
    print(f"Calling model with {len(req.messages)} messages")

@after_model
def log_response(resp):
    print(f"Model responded: {resp.content[:100]}")

@wrap_tool_call
def handle_errors(tool_call, execute):
    try:
        return execute(tool_call)
    except Exception as e:
        return f"Tool error: {e}"

agent = create_agent(
    "openai:gpt-4o",
    tools=[search_tool],
    system_prompt="Fallback prompt",
    middleware=[build_prompt, log_request, log_response, handle_errors],
)
```

### Class-based

```python
from langchain.agents import AgentMiddleware

class CustomMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        pass
    def after_model(self, state, runtime):
        pass
```

### Built-in Middleware

- **`SummarizationMiddleware`** — condenses message history approaching context limits
- **`HumanInTheLoopMiddleware`** — pauses for user approval before tool calls

---

## Model String Identifiers

Instead of instantiating `ChatOpenAI(model="gpt-4o")`, you can pass a string:

```python
agent = create_agent("openai:gpt-4o", tools)
agent = create_agent("google_genai:gemini-2.0-flash", tools)
agent = create_agent("groq:llama-3.3-70b", tools)
```

You can still pass `BaseChatModel` instances (e.g. from `llm_factory.build_llm()`).

---

## Structured Output

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    recommendations: list[str]

agent = create_agent(
    model,
    tools,
    system_prompt="Analyze the data.",
    response_format=AnalysisResult,
)
```

---

## Tools

Tool definition is unchanged — use `@tool` from `langchain_core.tools`:

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return do_search(query)
```

`ToolNode` from `langgraph.prebuilt` still works but tools now auto-validate (no need for `ValidationNode`).

---

## Invocation (unchanged)

```python
from langchain_core.messages import HumanMessage

# Sync
result = agent.invoke({"messages": [HumanMessage(content="What is X?")]})
final_message = result["messages"][-1].content

# Streaming
for event in agent.stream({"messages": [HumanMessage(content="What is X?")]}):
    print(event)

# Async
async for event in agent.astream({"messages": [HumanMessage(content="What is X?")]}):
    print(event)
```

---

## Migration Checklist for This Codebase

### Files to update

1. **`agent/graph.py`** — Import + `prompt=` → `system_prompt=`
2. **`insights/loop.py`** (lines ~110, ~591, ~676) — Import + `prompt=` → `system_prompt=`
3. **`tests/test_agent_e2e.py`** — Import + `prompt=` → `system_prompt=`
4. **`tests/test_insights.py`** — Patch targets: `"insights.loop.create_react_agent"` → `"insights.loop.create_agent"`

### Steps

1. Ensure deps: `uv add langchain>=1.0 langgraph>=1.0`
2. Update imports: `from langgraph.prebuilt import create_react_agent` → `from langchain.agents import create_agent`
3. Rename parameter: `prompt=` → `system_prompt=`
4. Update test patches from `create_react_agent` to `create_agent`
5. If streaming code checks for `"agent"` node name, change to `"model"`
6. Run tests: `make test`

---

## Custom State

If you need custom state beyond messages:

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class MyState(TypedDict):
    messages: list[BaseMessage]  # required
    context: str
    iteration: int

agent = create_agent(model, tools, system_prompt="...", state_schema=MyState)
```

---

## Multi-Agent Systems

```python
research_agent = create_agent(model, [search], system_prompt="Research agent", name="researcher")
writer_agent = create_agent(model, [], system_prompt="Writer agent", name="writer")
```

Agents can be composed in a LangGraph `StateGraph` as nodes.

---

## Package Versions (as of Feb 2026)

| Package | Version | Python |
|---------|---------|--------|
| `langgraph` | 1.0.9 | >=3.10 |
| `langchain` | 1.2.10 | >=3.10 |
| `langchain-core` | latest | >=3.10 |

---

## Common Patterns in This Codebase

### Agent with RAG tools (insights module)

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

tools = [rag_search_tool, rag_get_doc_tool]
agent = create_agent(self.llm, tools, system_prompt=AGENT_SYSTEM_PROMPT)
result = agent.invoke({"messages": [HumanMessage(content=query)]})
```

### Simple agent with web search (agent module)

```python
from langchain.agents import create_agent

def build_agent(model_name=None, system_message=DEFAULT_SYSTEM, provider="gemini"):
    llm = build_llm(provider=provider, model=model_name)
    return create_agent(llm, [brave_search], system_prompt=system_message)
```
