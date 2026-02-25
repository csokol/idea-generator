"""CLI entrypoint for the LangGraph agent."""

from __future__ import annotations

import argparse
import sys

from agent.graph import build_agent


def run_single(agent, query: str) -> None:
    result = agent.invoke({"messages": [("user", query)]})
    print(result["messages"][-1].content)


def run_interactive(agent) -> None:
    print("Agent REPL (type 'quit' or 'exit' to stop)")
    messages = []
    while True:
        try:
            query = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        messages.append(("user", query))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"\n{messages[-1].content}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="agent", description="AI agent with Brave Search")
    parser.add_argument("query", nargs="?", default=None, help="Question to answer (omit for interactive mode)")
    parser.add_argument("--provider", choices=["groq", "gemini", "openai"], default="gemini", help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None, help="LLM model name (default depends on provider)")
    args = parser.parse_args()

    agent = build_agent(model_name=args.model, provider=args.provider)

    if args.query:
        run_single(agent, args.query)
    else:
        run_interactive(agent)

    from token_tracker import get_tracker

    tracker = get_tracker()
    if tracker.has_usage():
        print(f"\n{tracker.summary()}")


if __name__ == "__main__":
    main()
