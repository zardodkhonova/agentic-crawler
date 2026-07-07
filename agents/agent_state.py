"""LangGraph agent state container."""

from __future__ import annotations

from typing import Any, TypedDict

from agents.schemas import Chunk, Plan, ValidationResult


class AgentState(TypedDict, total=False):
    """Mutable state passed through the LangGraph nodes."""

    original_query: str
    plan: Plan | None
    frontier: list[Any]
    visited_urls: set[str]
    indexed_chunk_ids: set[str]
    iteration: int
    draft_answer: str
    sources: list[Chunk]
    confidence: float
    unsupported_claims: list[Any]
    trace: list[dict[str, Any]]
    crawl_stats: dict[str, Any]
    validation_result: ValidationResult | None
    route: str
    pages_crawled: int
    latest_document: Any


def initial_state(query: str) -> AgentState:
    """Create a fresh agent state for a new query."""
    return AgentState(
        original_query=query,
        plan=None,
        frontier=[],
        visited_urls=set(),
        indexed_chunk_ids=set(),
        iteration=0,
        draft_answer="",
        sources=[],
        confidence=0.0,
        unsupported_claims=[],
        trace=[],
        crawl_stats={},
        validation_result=None,
        route="continue",
        pages_crawled=0,
        latest_document=None,
    )


def add_trace_entry(state: AgentState, node: str, message: str, **extra: Any) -> list[dict[str, Any]]:
    """Return an updated trace list with a new entry appended."""
    entry = {"node": node, "message": message, **extra}
    trace = list(state.get("trace", []))
    trace.append(entry)
    return trace
