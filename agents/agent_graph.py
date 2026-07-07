"""LangGraph Observe-Plan-Act-Validate agent."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.agent_state import AgentState, add_trace_entry, initial_state
from agents.schemas import AnswerResult, Chunk, Plan
from config.settings import Settings
from crawler.html_extractor import HtmlExtractor

logger = logging.getLogger(__name__)


def _chunk_to_agent(chunk) -> Chunk:
    return Chunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_url=chunk.source_url,
        chunk_index=chunk.chunk_index,
        crawled_at=chunk.crawled_at,
        relevance_score_at_crawl=chunk.relevance_score_at_crawl,
    )


def _plan_to_crawler(plan: Plan) -> Plan:
    return Plan(
        sub_queries=plan.sub_queries,
        seed_search_terms=plan.seed_search_terms,
        stop_condition_description=plan.stop_condition_description,
    )


def build_agent_graph(
    planner,
    crawler,
    indexer,
    retriever,
    validator,
    rag_engine,
    settings: Settings,
):
    """Build and compile the LangGraph agent loop.

    The loop follows the Observe -> Plan -> Act -> Validate cycle:
    - Observe: parse the most recently fetched page/content and update crawl state.
      On the very first cycle there is nothing to observe yet, so this step is a no-op.
    - Plan: create the initial plan (first cycle) or refine the plan based on
      unsupported claims from the previous Validate step. The iteration counter
      is advanced on every pass through this node, regardless of which branch runs,
      so the loop always makes progress toward max_iterations.
    - Act: fetch and index the next highest-priority page from the crawl frontier.
    - Validate: generate a draft answer, check claim support, compute confidence,
      and route back to Observe (continue) or to Finalize (accept / budget exhausted).
    """
    extractor = HtmlExtractor()
    max_iterations = settings.max_iterations
    confidence_threshold = settings.confidence_threshold

    async def observe_node(state: AgentState) -> dict[str, Any]:
        """Parse latest fetch results and update crawl statistics.

        On the first cycle, state has no plan and no latest_document yet, so this
        step simply reports empty/starting statistics and passes through to Plan.
        """
        trace = add_trace_entry(state, "observe", "Observing crawl state")
        stats = crawler.stats()
        updates: dict[str, Any] = {
            "crawl_stats": stats,
            "pages_crawled": stats.get("pages_crawled", state.get("pages_crawled", 0)),
            "trace": trace,
        }
        if state.get("latest_document") is not None:
            doc = state["latest_document"]
            visited = set(state.get("visited_urls", set()))
            visited.add(doc.url)
            updates["visited_urls"] = visited
        return updates

    async def plan_node(state: AgentState) -> dict[str, Any]:
        """Create initial plan or refine based on unsupported claims.

        The iteration counter is incremented on every pass through this node
        after the initial plan, whether or not a refinement actually happens,
        so the loop is guaranteed to reach max_iterations and terminate even
        when there are no unsupported claims to refine (e.g. an empty
        "I don't know" answer produces zero claims, which must still count
        as a completed iteration).
        """
        if state.get("plan") is None:
            trace = add_trace_entry(state, "plan", "Creating initial plan")
            plan = await planner.plan(state["original_query"])
            await crawler.seed(_plan_to_crawler(plan))
            return {"plan": plan, "iteration": 1, "trace": trace}

        next_iteration = state.get("iteration", 0) + 1

        if state.get("unsupported_claims") and state.get("route") == "continue":
            trace = add_trace_entry(state, "plan", "Refining plan for unsupported claims")
            follow_up = await planner.refine(
                state["original_query"],
                state["unsupported_claims"],
            )
            current_plan = state["plan"]
            refined_plan = Plan(
                sub_queries=current_plan.sub_queries + [follow_up],
                seed_search_terms=[follow_up],
                stop_condition_description=current_plan.stop_condition_description,
            )
            await crawler.seed(_plan_to_crawler(refined_plan))
            return {
                "plan": refined_plan,
                "iteration": next_iteration,
                "route": "continue",
                "trace": trace,
            }

        trace = add_trace_entry(
            state,
            "plan",
            "No unsupported claims to refine; continuing crawl with existing plan",
        )
        return {"iteration": next_iteration, "trace": trace}

    async def act_node(state: AgentState) -> dict[str, Any]:
        """Fetch and index the next highest-priority page."""
        trace = add_trace_entry(state, "act", "Crawling next page")
        document = await crawler.step()
        if document is None:
            return {"latest_document": None, "trace": trace}

        clean_text, metadata = extractor.extract(document.raw_html, document.url)
        chunks = indexer.process_and_store(document, (clean_text, metadata), 1.0)
        indexed_ids = set(state.get("indexed_chunk_ids", set()))
        for chunk in chunks:
            indexed_ids.add(chunk.chunk_id)

        return {
            "latest_document": document,
            "indexed_chunk_ids": indexed_ids,
            "pages_crawled": crawler.stats().get("pages_crawled", 0),
            "trace": trace,
        }

    async def validate_node(state: AgentState) -> dict[str, Any]:
        """Generate draft answer and validate claims against sources."""
        from validation.schemas import SourceChunk

        trace = add_trace_entry(state, "validate", "Generating and validating answer")

        plan = state.get("plan")
        sub_queries = list(getattr(plan, "sub_queries", []) or []) if plan else []

        answer_text, cited_chunks, all_retrieved_chunks = await rag_engine.answer(
            state["original_query"],
            sub_queries=sub_queries,
        )

        # Validate against every chunk that was actually shown to the LLM in
        # its context, not only the ones it happened to cite -- this avoids
        # penalizing correct answers where the model made a citation
        # formatting mistake or simply omitted a citation for a true claim.
        source_pool = all_retrieved_chunks or cited_chunks
        if not source_pool:
            source_pool = retriever.search(state["original_query"], settings.top_k)

        source_chunks = [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_url=chunk.source_url,
            )
            for chunk in source_pool
        ]

        validation = await validator.validate(answer_text, source_chunks)
        agent_sources = [_chunk_to_agent(chunk) for chunk in (cited_chunks or all_retrieved_chunks)]
        stats_pages = crawler.stats().get("pages_crawled", 0)
        iteration = state.get("iteration", 0)

        route: Literal["accept", "continue", "stop_budget_exhausted"] = "continue"
        normalized_answer = answer_text.strip().lower()
        is_idk = normalized_answer.startswith("i don't know") or normalized_answer.startswith(
            "i do not know"
        )
        crawl_stats = crawler.stats()
        frontier_empty = crawl_stats.get("frontier_size", 0) == 0

        if validation.confidence >= confidence_threshold:
            route = "accept"
        elif is_idk and (frontier_empty or iteration >= max_iterations):
            route = "stop_budget_exhausted"
        elif iteration >= max_iterations:
            route = "stop_budget_exhausted"
        elif stats_pages >= settings.max_pages_per_query:
            route = "stop_budget_exhausted"

        trace = add_trace_entry(
            state,
            "validate",
            f"Confidence {validation.confidence:.2f}, route={route}",
            confidence=validation.confidence,
        )
        return {
            "draft_answer": answer_text,
            "sources": agent_sources,
            "confidence": validation.confidence,
            "unsupported_claims": validation.unsupported_claims,
            "validation_result": validation,
            "route": route,
            "trace": trace,
        }

    async def finalize_node(state: AgentState) -> dict[str, Any]:
        """Finalize the agent run."""
        trace = add_trace_entry(
            state,
            "finalize",
            "Run complete",
            route=state.get("route", "accept"),
        )
        return {"trace": trace}

    def route_after_validate(state: AgentState) -> str:
        route = state.get("route", "continue")
        if route in ("accept", "stop_budget_exhausted"):
            return "finalize"
        if state.get("iteration", 0) >= max_iterations:
            return "finalize"
        if crawler.stats().get("pages_crawled", 0) >= settings.max_pages_per_query:
            return "finalize"
        return "observe"

    graph = StateGraph(AgentState)
    graph.add_node("observe", observe_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("observe")
    graph.add_edge("observe", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"observe": "observe", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


class AgenticCrawler:
    """High-level wrapper around the compiled LangGraph agent."""

    def __init__(
        self,
        compiled_graph,
        settings: Settings,
        crawler,
        vector_store=None,
    ) -> None:
        self._graph = compiled_graph
        self._settings = settings
        self._crawler = crawler
        self._vector_store = vector_store

    def reset(self) -> None:
        """Reset crawler and vector index before starting a new, independent query.

        Must be called between separate queries (e.g. successive benchmark
        questions) so that one question's crawl frontier, dedup tracker,
        page counters, and indexed chunks do not leak into the next.
        """
        self._crawler.reset()
        if self._vector_store is not None:
            self._vector_store.reset()

    async def run(self, query: str) -> AnswerResult:
        """Execute the agent loop for a user query."""
        try:
            final_state = await self._graph.ainvoke(initial_state(query))
        except Exception as exc:
            logger.error("Agent run failed: %s", exc)
            raise

        stats = self._crawler.stats()
        return AnswerResult(
            answer_text=final_state.get("draft_answer") or "I don't know.",
            sources=final_state.get("sources", []),
            confidence=float(final_state.get("confidence", 0.0)),
            iterations_used=int(final_state.get("iteration", 0)),
            pages_crawled=int(stats.get("pages_crawled", 0)),
            trace=final_state.get("trace", []),
        )