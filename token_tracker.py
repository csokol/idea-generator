"""Token usage tracking and cost estimation for LLM calls."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Approximate costs per 1M tokens (input, output) in USD.
COST_TABLE: dict[str, tuple[float, float]] = {
    "qwen/qwen3-32b": (0.30, 0.30),
    "gemini-3-flash-preview": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-mini": (0.80, 3.20),
}
_DEFAULT_COST = (1.00, 3.00)


class TokenTracker:
    """Accumulates token usage per (provider, model) pair."""

    def __init__(self) -> None:
        self._usage: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        key = (provider, model)
        self._usage[key][0] += input_tokens
        self._usage[key][1] += output_tokens

    def has_usage(self) -> bool:
        return any(v[0] > 0 or v[1] > 0 for v in self._usage.values())

    def total_cost(self) -> float:
        total = 0.0
        for (_, model), (inp, out) in self._usage.items():
            ci, co = COST_TABLE.get(model, _DEFAULT_COST)
            total += inp * ci / 1_000_000 + out * co / 1_000_000
        return total

    def summary(self) -> str:
        if not self.has_usage():
            return ""

        lines = [
            "\u2500\u2500\u2500 Token Usage \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"{'Provider/Model':<30} {'Input':>10} {'Output':>10} {'Est. Cost':>10}",
        ]

        total_in = 0
        total_out = 0
        total_cost = 0.0

        for (provider, model), (inp, out) in sorted(self._usage.items()):
            ci, co = COST_TABLE.get(model, _DEFAULT_COST)
            cost = inp * ci / 1_000_000 + out * co / 1_000_000
            label = f"{provider}/{model}"
            if len(label) > 30:
                label = label[:27] + "..."
            lines.append(f"{label:<30} {inp:>10,} {out:>10,} ${cost:>8.3f}")
            total_in += inp
            total_out += out
            total_cost += cost

        lines.append("\u2500" * 63)
        lines.append(f"{'Total':<30} {total_in:>10,} {total_out:>10,} ${total_cost:>8.3f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return structured token usage data for report integration."""
        rows = []
        total_in = 0
        total_out = 0
        total_cost = 0.0

        for (provider, model), (inp, out) in sorted(self._usage.items()):
            ci, co = COST_TABLE.get(model, _DEFAULT_COST)
            cost = inp * ci / 1_000_000 + out * co / 1_000_000
            rows.append({
                "provider": provider,
                "model": model,
                "input_tokens": inp,
                "output_tokens": out,
                "cost": cost,
            })
            total_in += inp
            total_out += out
            total_cost += cost

        return {
            "rows": rows,
            "total_input": total_in,
            "total_output": total_out,
            "total_cost": total_cost,
        }


class TokenTrackingHandler(BaseCallbackHandler):
    """LangChain callback handler that records token usage into a TokenTracker."""

    def __init__(self, tracker: TokenTracker, provider: str, model: str) -> None:
        super().__init__()
        self.tracker = tracker
        self.provider = provider
        self.model = model

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        usage = {}
        # Try llm_output first (OpenAI, Groq style)
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if not usage:
                usage = response.llm_output.get("usage", {})

        # Try usage_metadata from generations (Gemini style)
        if not usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen, "generation_info", {}) or {}
                    um = meta.get("usage_metadata", {})
                    if um:
                        usage = {
                            "prompt_tokens": um.get("prompt_token_count", 0) or um.get("input_tokens", 0),
                            "completion_tokens": um.get("candidates_token_count", 0) or um.get("output_tokens", 0),
                        }
                        break
                if usage:
                    break

        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        if input_tokens or output_tokens:
            self.tracker.record(self.provider, self.model, input_tokens, output_tokens)


# Module-level singleton
_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker
