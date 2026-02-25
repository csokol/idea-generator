"""Insight agent orchestrator — iterative RAG search, signal extraction, clustering."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

log = logging.getLogger(__name__)


def _format_messages(messages) -> str:
    """Format a list of LangChain message objects into readable text."""
    parts = []
    for msg in messages:
        if hasattr(msg, "type"):
            role = msg.type  # "system", "human", "ai", etc.
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            role, msg = msg[0], type("M", (), {"content": msg[1]})()
        else:
            role = type(msg).__name__
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            content = "".join(
                b if isinstance(b, str) else b.get("text", "")
                for b in content
            )
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class _LLMLoggingHandler(BaseCallbackHandler):
    """LangChain callback that logs LLM call inputs/outputs to a file."""

    def __init__(self, log_file: str | None = None) -> None:
        self._starts: dict[str, float] = {}
        self._log_file = log_file
        self._call_num = 0

    def _write(self, text: str) -> None:
        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(text)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        model = serialized.get("kwargs", {}).get("model", "") or serialized.get("id", [""])[-1]
        log.info("[LLM START] agent call model=%s", model)
        self._starts[str(run_id)] = time.monotonic()
        self._call_num += 1
        self._write(
            f"\n{'='*80}\n"
            f"[CALL #{self._call_num}] LLM START  model={model}\n"
            f"{'='*80}\n\n"
            f"--- PROMPTS ---\n"
            + "\n---\n".join(prompts)
            + "\n\n"
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        model = serialized.get("kwargs", {}).get("model", "") or serialized.get("id", [""])[-1]
        log.info("[LLM START] agent call model=%s", model)
        self._starts[str(run_id)] = time.monotonic()
        self._call_num += 1
        # messages is a list of lists of BaseMessage
        formatted = ""
        for msg_list in messages:
            formatted += _format_messages(msg_list) + "\n"
        self._write(
            f"\n{'='*80}\n"
            f"[CALL #{self._call_num}] CHAT MODEL START  model={model}\n"
            f"{'='*80}\n\n"
            f"--- INPUT MESSAGES ---\n{formatted}\n"
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        t0 = self._starts.pop(str(run_id), None)
        elapsed_s = time.monotonic() - t0 if t0 is not None else 0
        elapsed = f" ({elapsed_s:.1f}s)" if t0 is not None else ""
        tokens = ""
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage", {})
            if usage:
                tokens = f" tokens={usage}"
        log.info("[LLM DONE] agent call%s%s", elapsed, tokens)
        # Extract output text
        output_text = ""
        if hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    output_text += getattr(gen, "text", str(gen.message.content if hasattr(gen, "message") else gen)) + "\n"
        self._write(
            f"--- OUTPUT ({elapsed_s:.1f}s) ---\n{output_text}\n"
        )

_MAX_RETRIES = 5
_INITIAL_WAIT = 10  # seconds
_MAX_WAIT = 120  # seconds


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for 429 / RESOURCE_EXHAUSTED errors from LLM providers."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate" in msg

from langgraph.prebuilt import create_react_agent

from insights.models import EvidenceStore, Opportunity, Signal
from insights.prompts import (
    AGENT_ITERATION_PROMPT,
    AGENT_SYSTEM_PROMPT,
    AUTONOMOUS_ITERATION_PROMPT,
    AUTONOMOUS_SYSTEM_PROMPT,
    CLUSTER_MERGE_PROMPT,
    EVALUATE_PROGRESS_PROMPT,
    EXTRACT_SIGNALS_PROMPT,
    MARKET_VALIDATION_PROMPT,
    PLAN_QUERIES_PROMPT,
)
from insights.rag_store import RagStore


def _content_to_str(content: Any) -> str:
    """Normalise LLM response content to a plain string.

    Some providers return a list of content blocks instead of a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return str(content)


def _parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    cleaned = _content_to_str(text).strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def _fuzzy_match_key(key: str, valid_keys: set[str]) -> str | None:
    """Find the best matching valid key for a given key, or None."""
    # Exact match
    if key in valid_keys:
        return key
    # Case-insensitive match
    lower = key.lower().strip()
    for vk in valid_keys:
        if vk.lower().strip() == lower:
            return vk
    # Substring containment (either direction)
    for vk in valid_keys:
        if lower in vk.lower() or vk.lower() in lower:
            return vk
    return None


class InsightLoop:
    def __init__(
        self,
        goal: str,
        rag_store: RagStore,
        llm: BaseChatModel,
        *,
        iterations: int | None = None,
        max_iterations: int = 3,
        top_k: int = 10,
        queries_per_iteration: int = 5,
        max_evidence: int = 200,
        min_evidence_per_opportunity: int = 2,
        target_opportunities: int | None = None,
        goal_description: str | None = None,
        use_agent: bool = True,
        brave_token: str | None = None,
        autonomous: bool = False,
        pg_conn=None,
        index_table: str = "rag_chunks",
        index_namespace: str = "",
        index_dimension: int = 1536,
        embedder=None,
        llm_log_file: str | None = None,
    ) -> None:
        self.goal = goal
        self.rag_store = rag_store
        self.llm = llm
        # Accept `iterations` for backward compat, prefer `max_iterations`
        self.max_iterations = iterations if iterations is not None else max_iterations
        self.top_k = top_k
        self.queries_per_iteration = queries_per_iteration
        self.max_evidence = max_evidence
        self.min_evidence_per_opportunity = min_evidence_per_opportunity
        self.target_opportunities = target_opportunities
        self.goal_description = goal_description
        self.use_agent = use_agent
        self.brave_token = brave_token
        self.autonomous = autonomous
        self.pg_conn = pg_conn
        self.index_table = index_table
        self.index_namespace = index_namespace
        self.index_dimension = index_dimension
        self.embedder = embedder
        self.llm_log_file = llm_log_file
        self.evidence = EvidenceStore()
        self.all_queries: list[str] = []
        self.seen_doc_ids: set[str] = set()
        self._suggested_focus: str | None = None
        self.indexed_urls: set[str] = set()

    def _write_llm_log(self, text: str) -> None:
        if self.llm_log_file:
            with open(self.llm_log_file, "a") as f:
                f.write(text)

    def _llm_invoke(self, messages: list[BaseMessage], *, label: str = "llm_invoke") -> Any:
        """Invoke LLM with retry + exponential backoff on rate-limit errors."""
        log.info("[LLM START] %s", label)
        self._write_llm_log(
            f"\n{'='*80}\n"
            f"[{label}] LLM START\n"
            f"{'='*80}\n\n"
            f"--- INPUT ---\n{_format_messages(messages)}\n\n"
        )
        t0 = time.monotonic()
        wait = _INITIAL_WAIT
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self.llm.invoke(messages)
                elapsed = time.monotonic() - t0
                log.info("[LLM DONE] %s (%.1fs)", label, elapsed)
                self._write_llm_log(
                    f"--- OUTPUT ({elapsed:.1f}s) ---\n"
                    f"{_content_to_str(result.content)}\n\n"
                )
                return result
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt == _MAX_RETRIES:
                    elapsed = time.monotonic() - t0
                    log.error("[LLM FAIL] %s after %.1fs: %s", label, elapsed, exc)
                    raise
                log.warning(
                    "Rate-limited, retrying in %.0fs (attempt %d/%d)...",
                    wait, attempt, _MAX_RETRIES,
                )
                time.sleep(wait)
                wait = min(wait * 2, _MAX_WAIT)

    def run(self) -> list[Opportunity]:
        if self.autonomous:
            return self._run_autonomous_loop()
        if self.use_agent:
            return self._run_agent_loop()
        return self._run_manual_loop()

    def _run_manual_loop(self) -> list[Opportunity]:
        for iteration in range(1, self.max_iterations + 1):
            log.info("=== Iteration %d/%d ===", iteration, self.max_iterations)

            queries = self._plan_queries(iteration)
            log.info("Planned %d queries", len(queries))
            for q in queries:
                log.info("  query: %s", q)

            docs = self._retrieve(queries)
            new_docs = [d for d in docs if d["doc_id"] not in self.seen_doc_ids]
            self.seen_doc_ids.update(d["doc_id"] for d in new_docs)
            log.info("Retrieved %d docs (%d new)", len(docs), len(new_docs))

            if new_docs:
                log.info("Extracting signals from %d documents...", len(new_docs))
                signals = self._extract_signals(new_docs)
                added = sum(1 for s in signals if self.evidence.add(s))
                log.info("Extracted %d signals (%d added after dedup)", len(signals), added)

            unique_urls = {s.url for g in self.evidence.pain_groups.values() for s in g}
            log.info(
                "Iteration %d summary: %d total signals, %d pain groups, %d unique URLs",
                iteration,
                self.evidence.total_signals,
                len(self.evidence.pain_groups),
                len(unique_urls),
            )

            # --- Structured stop checks (override LLM) ---
            if self.evidence.total_signals >= self.max_evidence:
                log.info("Max evidence reached, stopping.")
                break

            if iteration > 1 and not new_docs:
                log.info("No new documents found, stopping.")
                break

            if iteration == self.max_iterations:
                log.info("Max iterations reached, stopping.")
                break

            # --- Agentic evaluation ---
            should_continue = self._evaluate_progress(iteration)
            if not should_continue:
                log.info("Agent decided to stop after iteration %d.", iteration)
                break

        log.info("Clustering and merging...")
        opportunities = self._cluster_and_merge()
        opportunities = self._score_opportunities(opportunities)
        self._validate_market(opportunities)
        return opportunities

    def _evaluate_progress(self, iteration: int) -> bool:
        """Ask LLM whether to continue, with structured overrides."""
        # Lightweight clustering to see current opportunity count
        candidate_opps = self._cluster_and_merge()

        # Structured check: target opportunities met
        if self.target_opportunities is not None:
            qualified = [o for o in candidate_opps if len(o.signals) >= self.min_evidence_per_opportunity]
            if len(qualified) >= self.target_opportunities:
                log.info(
                    "Target met: %d/%d qualified opportunities found, stopping.",
                    len(qualified), self.target_opportunities,
                )
                return False

        unique_urls = {s.url for g in self.evidence.pain_groups.values() for s in g}
        opp_summaries = "\n".join(
            f"- {o.title} ({len(o.signals)} signals, confidence={o.confidence})"
            for o in candidate_opps
        ) if candidate_opps else "(none yet)"

        goal_desc_block = ""
        if self.goal_description:
            goal_desc_block = f"\nDetailed goal: {self.goal_description}\n"

        prompt = EVALUATE_PROGRESS_PROMPT.format(
            goal=self.goal,
            goal_description_block=goal_desc_block,
            iteration=iteration,
            max_iterations=self.max_iterations,
            total_signals=self.evidence.total_signals,
            num_pain_groups=len(self.evidence.pain_groups),
            num_opportunities=len(candidate_opps),
            num_unique_urls=len(unique_urls),
            opportunity_summaries=opp_summaries,
        )
        try:
            response = self._llm_invoke([HumanMessage(content=prompt)], label=f"evaluate_progress (iteration {iteration})")
            result = _parse_json_response(response.content)
            should_continue = result.get("should_continue", True)
            reasoning = result.get("reasoning", "")
            self._suggested_focus = result.get("suggested_focus")
            log.info(
                "Agent evaluation: should_continue=%s, reasoning=%s",
                should_continue, reasoning,
            )
            return bool(should_continue)
        except (json.JSONDecodeError, TypeError, KeyError, Exception) as exc:
            log.warning("Failed to parse evaluate_progress response: %s", exc)
            # Default to continuing on parse failure
            return True

    def _plan_queries(self, iteration: int) -> list[str]:
        themes = list(self.evidence.pain_groups.keys())[:20]
        focus_block = ""
        if self._suggested_focus:
            focus_block = f"\nSuggested focus for this iteration: {self._suggested_focus}\n"
        prompt = PLAN_QUERIES_PROMPT.format(
            iteration=iteration,
            goal=self.goal,
            current_themes="\n".join(f"- {t}" for t in themes) if themes else "(none yet)",
            previous_queries="\n".join(f"- {q}" for q in self.all_queries[-20:]) if self.all_queries else "(none)",
            num_queries=self.queries_per_iteration,
            suggested_focus_block=focus_block,
        )
        log.debug("LLM plan_queries prompt:\n%s", prompt)
        response = self._llm_invoke([
            SystemMessage(content="You generate search queries as JSON arrays."),
            HumanMessage(content=prompt),
        ], label=f"plan_queries (iteration {iteration})")
        log.debug("LLM plan_queries response:\n%s", response.content)
        try:
            queries = _parse_json_response(response.content)
            if not isinstance(queries, list):
                queries = [self.goal]
        except (json.JSONDecodeError, TypeError):
            log.warning("Failed to parse plan_queries response: %s", _content_to_str(response.content)[:200])
            queries = [self.goal]

        self.all_queries.extend(queries)
        return queries

    def _retrieve(self, queries: list[str]) -> list[dict]:
        """Search RAG store, then expand chunk hits into full documents."""
        # Phase 1: collect unique doc IDs from chunk-level search hits
        seen_doc_ids: set[str] = set()
        doc_order: list[str] = []
        domain_counts: dict[str, int] = {}
        for q in queries:
            hits = self.rag_store.search(q, top_k=self.top_k)
            new_for_query = 0
            for h in hits:
                doc_id = h["id"].split("#")[0]
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc_order.append(doc_id)
                    new_for_query += 1
                    domain = h.get("source_domain", "unknown")
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
            top_score = hits[0]["score"] if hits else 0
            log.info("  %r → %d hits (%d new docs, top=%.3f)", q, len(hits), new_for_query, top_score)
        if domain_counts:
            top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info("  Top domains: %s", ", ".join(f"{d}({n})" for d, n in top_domains))

        # Phase 2: fetch full documents for each unique doc_id
        documents: list[dict] = []
        for doc_id in doc_order:
            chunks = self.rag_store.get_doc(doc_id)
            if not chunks:
                continue
            first = chunks[0]
            full_text = "\n\n".join(c["chunk_text"] for c in chunks)
            chunk_ids = [c["id"] for c in chunks]
            documents.append({
                "doc_id": doc_id,
                "url": first.get("url", ""),
                "title": first.get("title", ""),
                "source_domain": first.get("source_domain", ""),
                "full_text": full_text,
                "chunk_ids": chunk_ids,
            })
        log.info("Expanded %d doc IDs into %d documents", len(doc_order), len(documents))
        return documents

    def _extract_signals(self, docs: list[dict]) -> list[Signal]:
        # Process in batches — smaller batches since full pages are larger
        batch_size = 3
        num_batches = (len(docs) + batch_size - 1) // batch_size
        all_signals = []
        for i in range(0, len(docs), batch_size):
            batch_num = i // batch_size + 1
            if num_batches > 1:
                log.info("  Extracting batch %d/%d...", batch_num, num_batches)
            batch = docs[i : i + batch_size]
            chunks_text = "\n\n".join(
                f"[{d['doc_id']}] URL: {d['url']}\nTitle: {d.get('title', '')}\n{d['full_text']}"
                for d in batch
            )
            prompt = EXTRACT_SIGNALS_PROMPT.format(goal=self.goal, chunks_text=chunks_text)
            log.debug("LLM extract_signals prompt (%d chars):\n%s", len(prompt), prompt)
            response = self._llm_invoke([HumanMessage(content=prompt)], label=f"extract_signals (batch {batch_num}/{num_batches})")
            log.debug("LLM extract_signals response:\n%s", response.content)
            try:
                raw = _parse_json_response(response.content)
                if not isinstance(raw, list):
                    continue
                for item in raw:
                    sig = Signal(
                        pain=item.get("pain", ""),
                        workaround=item.get("workaround", "none mentioned"),
                        desired_outcome=item.get("desired_outcome", ""),
                        segment=item.get("segment", ""),
                        severity=int(item.get("severity", 3)),
                        willingness_to_pay=int(item.get("willingness_to_pay", 3)),
                        keywords=item.get("keywords", []),
                        doc_id=item.get("doc_id", ""),
                        url=item.get("url", ""),
                    )
                    log.info(
                        "  signal: %s | sev=%d wtp=%d seg=%s",
                        sig.pain[:80],
                        sig.severity,
                        sig.willingness_to_pay,
                        sig.segment,
                    )
                    all_signals.append(sig)
            except (json.JSONDecodeError, TypeError, KeyError):
                log.warning("Failed to parse extract_signals batch: %s", _content_to_str(response.content)[:200])
                continue
        return all_signals

    def _cluster_and_merge(self) -> list[Opportunity]:
        pain_groups = self.evidence.pain_groups
        if not pain_groups:
            return []

        valid_keys = set(pain_groups.keys())
        groups_text = "\n\n".join(
            f"Group '{key}' ({len(signals)} signals):\n"
            + "\n".join(f"  - {s.pain} (severity={s.severity}, wtp={s.willingness_to_pay})" for s in signals[:5])
            for key, signals in pain_groups.items()
        )
        keys_list = "\n".join(f"- {k}" for k in valid_keys)
        prompt = CLUSTER_MERGE_PROMPT.format(pain_groups=groups_text, valid_keys=keys_list)
        log.debug("LLM cluster_merge prompt (%d chars):\n%s", len(prompt), prompt)
        response = self._llm_invoke([HumanMessage(content=prompt)], label="cluster_and_merge")
        log.debug("LLM cluster_merge response:\n%s", response.content)

        try:
            clusters = _parse_json_response(response.content)
            if not isinstance(clusters, list):
                return self._fallback_opportunities()
        except (json.JSONDecodeError, TypeError):
            log.warning("Failed to parse cluster response: %s", _content_to_str(response.content)[:200])
            return self._fallback_opportunities()

        opportunities = []
        for cluster in clusters:
            pain_keys = cluster.get("pain_keys", [])
            signals = []
            for key in pain_keys:
                matched = _fuzzy_match_key(key, valid_keys)
                if matched:
                    signals.extend(pain_groups.get(matched, []))
                else:
                    log.warning("Cluster pain_key %r not matched to any valid key", key)
            title = cluster.get("title", "Untitled")
            confidence = cluster.get("confidence", "low")
            log.info(
                "  cluster: %s | %d pain keys, %d signals, confidence=%s",
                title,
                len(pain_keys),
                len(signals),
                confidence,
            )
            if len(signals) < self.min_evidence_per_opportunity:
                log.info("    skipped (below min evidence %d)", self.min_evidence_per_opportunity)
                continue
            opportunities.append(Opportunity(
                title=title,
                summary=cluster.get("summary", ""),
                signals=signals,
                confidence=confidence,
            ))
        return opportunities

    def _fallback_opportunities(self) -> list[Opportunity]:
        """Create one opportunity per pain group when LLM clustering fails."""
        opportunities = []
        for key, signals in self.evidence.pain_groups.items():
            if len(signals) < self.min_evidence_per_opportunity:
                continue
            opportunities.append(Opportunity(
                title=key.title(),
                summary=signals[0].pain,
                signals=signals,
                confidence="low",
            ))
        return opportunities

    def _run_agent_loop(self) -> list[Opportunity]:
        from insights.tools import (
            make_fetch_page_tool,
            make_get_progress_tool,
            make_rag_get_doc_tool,
            make_rag_search_tool,
            make_record_signal_tool,
            make_web_search_tool,
        )

        for iteration in range(1, self.max_iterations + 1):
            log.info("=== Agent Iteration %d/%d ===", iteration, self.max_iterations)

            signals_before = self.evidence.total_signals

            # Build tools (fresh closures sharing self.evidence / self.rag_store)
            tools = [
                make_rag_search_tool(self.rag_store, top_k_default=self.top_k),
                make_rag_get_doc_tool(self.rag_store),
                make_record_signal_tool(self.evidence),
                make_get_progress_tool(self.evidence, self.max_evidence),
            ]
            if self.brave_token:
                tools.append(make_web_search_tool(self.brave_token))
                tools.append(make_fetch_page_tool())

            # Create a fresh agent per iteration
            agent = create_react_agent(self.llm, tools, prompt=AGENT_SYSTEM_PROMPT)

            # Build user message with progress context
            user_message = self._build_agent_iteration_message(iteration)

            log.info("Invoking agent for iteration %d...", iteration)
            t0 = time.monotonic()
            result = agent.invoke(
                {"messages": [("user", user_message)]},
                config={"recursion_limit": 50, "callbacks": [_LLMLoggingHandler(log_file=self.llm_log_file)]},
            )
            elapsed = time.monotonic() - t0
            log.info("Agent iteration %d completed in %.1fs", iteration, elapsed)

            # Extract rag_search queries from tool call history
            agent_messages = result.get("messages", [])
            self._extract_queries_from_messages(agent_messages)
            self._log_agent_messages(agent_messages)

            signals_added = self.evidence.total_signals - signals_before
            log.info(
                "Iteration %d: %d new signals, %d total signals, %d pain groups",
                iteration,
                signals_added,
                self.evidence.total_signals,
                len(self.evidence.pain_groups),
            )

            # Stop conditions
            if self.evidence.total_signals >= self.max_evidence:
                log.info("Max evidence reached (%d), stopping.", self.max_evidence)
                break

            if iteration > 1 and signals_added == 0:
                log.info("No new signals in iteration %d, stopping.", iteration)
                break

            if iteration == self.max_iterations:
                log.info("Max iterations reached, stopping.")
                break

        log.info("Clustering and merging...")
        opportunities = self._cluster_and_merge()
        opportunities = self._score_opportunities(opportunities)
        self._validate_market(opportunities)
        return opportunities

    def _run_autonomous_loop(self) -> list[Opportunity]:
        from index_rag.indexer import PgVectorIndexer

        from insights.tools import (
            make_fetch_page_tool,
            make_get_progress_tool,
            make_index_to_rag_tool,
            make_rag_get_doc_tool,
            make_rag_search_tool,
            make_record_signal_tool,
            make_web_search_tool,
        )

        # Ensure RAG table exists
        indexer = PgVectorIndexer(self.pg_conn, self.index_table, self.index_dimension)
        indexer.ensure_index()

        for iteration in range(1, self.max_iterations + 1):
            log.info("=== Autonomous Iteration %d/%d ===", iteration, self.max_iterations)

            signals_before = self.evidence.total_signals

            # Build tools (fresh closures sharing state across iterations)
            tools = [
                make_rag_search_tool(self.rag_store, top_k_default=self.top_k),
                make_rag_get_doc_tool(self.rag_store),
                make_record_signal_tool(self.evidence),
                make_get_progress_tool(self.evidence, self.max_evidence),
                make_index_to_rag_tool(
                    self.pg_conn, self.index_table, self.index_namespace,
                    self.embedder, self.index_dimension, self.goal,
                    indexed_urls=self.indexed_urls,
                ),
            ]
            if self.brave_token:
                tools.append(make_web_search_tool(self.brave_token))
                tools.append(make_fetch_page_tool())

            agent = create_react_agent(self.llm, tools, prompt=AUTONOMOUS_SYSTEM_PROMPT)

            user_message = self._build_autonomous_iteration_message(iteration)

            log.info("Invoking autonomous agent for iteration %d...", iteration)
            t0 = time.monotonic()
            result = agent.invoke(
                {"messages": [("user", user_message)]},
                config={"recursion_limit": 40, "callbacks": [_LLMLoggingHandler(log_file=self.llm_log_file)]},
            )
            elapsed = time.monotonic() - t0
            log.info("Autonomous iteration %d completed in %.1fs", iteration, elapsed)

            agent_messages = result.get("messages", [])
            self._extract_queries_from_messages(agent_messages)
            self._log_agent_messages(agent_messages)

            # Clear search cache so newly indexed content is searchable
            if hasattr(self.rag_store, '_search_cache'):
                self.rag_store._search_cache.clear()
            if hasattr(self.rag_store, '_embed_cache'):
                self.rag_store._embed_cache.clear()

            signals_added = self.evidence.total_signals - signals_before
            log.info(
                "Iteration %d: %d new signals, %d total signals, %d pain groups, %d indexed pages",
                iteration, signals_added, self.evidence.total_signals,
                len(self.evidence.pain_groups), len(self.indexed_urls),
            )

            # Stop conditions
            if self.evidence.total_signals >= self.max_evidence:
                log.info("Max evidence reached (%d), stopping.", self.max_evidence)
                break

            if iteration > 1 and signals_added == 0:
                log.info("No new signals in iteration %d, stopping.", iteration)
                break

            if iteration == self.max_iterations:
                log.info("Max iterations reached, stopping.")
                break

        log.info("Clustering and merging...")
        opportunities = self._cluster_and_merge()
        opportunities = self._score_opportunities(opportunities)
        self._validate_market(opportunities)
        return opportunities

    def _build_autonomous_iteration_message(self, iteration: int) -> str:
        progress = self._build_progress_summary()
        prev_queries = (
            "\n".join(f"- {q}" for q in self.all_queries[-20:])
            if self.all_queries
            else "(none yet)"
        )
        suggested_focus = ""
        if self._suggested_focus:
            suggested_focus = f"\nSuggested focus: {self._suggested_focus}\n"

        # Phase-aware instructions
        midpoint = max(self.max_iterations // 2, 1)
        if iteration <= midpoint:
            phase_instructions = (
                f"Focus on RESEARCHING and INDEXING: Use web_search to find {self.queries_per_iteration} "
                "different pages related to the goal. For each relevant result, fetch_page and then "
                "index_to_rag to build the corpus. Also start extracting signals from promising content."
            )
        else:
            phase_instructions = (
                f"Focus on ANALYZING and RECORDING: Use rag_search with {self.queries_per_iteration} "
                "varied queries to find pain points in your indexed corpus. Read full documents with "
                "rag_get_doc and record_signal for each distinct pain point. You can still index new "
                "pages if you find gaps."
            )

        return AUTONOMOUS_ITERATION_PROMPT.format(
            goal=self.goal,
            iteration=iteration,
            max_iterations=self.max_iterations,
            progress_summary=progress,
            indexed_count=len(self.indexed_urls),
            previous_queries=prev_queries,
            suggested_focus=suggested_focus,
            phase_instructions=phase_instructions,
        )

    def _build_agent_iteration_message(self, iteration: int) -> str:
        progress = self._build_progress_summary()
        prev_queries = (
            "\n".join(f"- {q}" for q in self.all_queries[-20:])
            if self.all_queries
            else "(none yet)"
        )
        suggested_focus = ""
        if self._suggested_focus:
            suggested_focus = f"\nSuggested focus: {self._suggested_focus}\n"
        return AGENT_ITERATION_PROMPT.format(
            goal=self.goal,
            iteration=iteration,
            max_iterations=self.max_iterations,
            progress_summary=progress,
            previous_queries=prev_queries,
            suggested_focus=suggested_focus,
            queries_to_try=self.queries_per_iteration,
        )

    def _build_progress_summary(self) -> str:
        total = self.evidence.total_signals
        groups = self.evidence.pain_groups
        top_themes = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        lines = [
            f"Total signals: {total}/{self.max_evidence}",
            f"Pain groups: {len(groups)}",
        ]
        if top_themes:
            lines.append("Top themes:")
            for key, signals in top_themes:
                lines.append(f"  - {key!r} ({len(signals)} signals)")
        else:
            lines.append("No signals recorded yet.")
        return "\n".join(lines)

    def _extract_queries_from_messages(self, messages: list) -> None:
        """Extract rag_search query args from agent message history."""
        for msg in messages:
            if not hasattr(msg, "tool_calls"):
                continue
            for tc in msg.tool_calls:
                if tc.get("name") == "rag_search":
                    query = tc.get("args", {}).get("query")
                    if query:
                        self.all_queries.append(query)

    def _log_agent_messages(self, messages: list) -> None:
        """Log a summary of LLM calls and tool calls from agent message history."""
        llm_calls = 0
        tool_calls: dict[str, int] = {}
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "ai":
                content = getattr(msg, "content", "")
                calls = getattr(msg, "tool_calls", [])
                if calls:
                    for tc in calls:
                        name = tc.get("name", "unknown")
                        tool_calls[name] = tool_calls.get(name, 0) + 1
                    llm_calls += 1
                elif content:
                    llm_calls += 1
        tool_summary = ", ".join(f"{n}={c}" for n, c in sorted(tool_calls.items()))
        log.info(
            "  Agent steps: %d LLM calls, tool calls: {%s}",
            llm_calls, tool_summary,
        )

    def _score_opportunities(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        for opp in opportunities:
            evidence_score = min(opp.evidence_count / 10.0, 1.0)
            severity_score = opp.avg_severity / 5.0
            wtp_score = opp.avg_willingness_to_pay / 5.0
            source_diversity = min(len(opp.unique_urls) / 5.0, 1.0)
            opp.score = (
                evidence_score * 0.3
                + severity_score * 0.3
                + wtp_score * 0.25
                + source_diversity * 0.15
            )
        opportunities.sort(key=lambda o: o.score, reverse=True)
        for rank, opp in enumerate(opportunities, 1):
            log.info(
                "  #%d %s (score=%.3f, evidence=%d, urls=%d)",
                rank,
                opp.title,
                opp.score,
                opp.evidence_count,
                len(opp.unique_urls),
            )
        return opportunities

    def _validate_market(self, opportunities: list[Opportunity]) -> None:
        """Search the web for each opportunity and ask LLM to assess existing competition."""
        if not self.brave_token:
            return
        from research_loop.search import BraveSearchClient
        client = BraveSearchClient(token=self.brave_token)
        for opp in opportunities:
            try:
                query = f"{opp.title} existing solutions alternatives"
                data = client.search(query, count=5)
                results = client.extract_results(data)
                search_results = "\n".join(
                    f"- {r.get('title', '')}: {r.get('url', '')}\n  {r.get('description', '')}"
                    for r in results
                ) if results else "(no results)"
                prompt = MARKET_VALIDATION_PROMPT.format(
                    title=opp.title,
                    summary=opp.summary,
                    search_results=search_results,
                )
                response = self._llm_invoke(
                    [HumanMessage(content=prompt)],
                    label=f"validate_market: {opp.title[:50]}",
                )
                opp.market_context = _content_to_str(response.content).strip()
            except Exception as exc:
                log.warning("Market validation failed for %r: %s", opp.title, exc)
