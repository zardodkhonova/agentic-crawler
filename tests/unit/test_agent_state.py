"""Tests for agent state."""

from __future__ import annotations

from agents.agent_state import add_trace_entry, initial_state


def test_initial_state_and_trace() -> None:
    state = initial_state("test query")
    assert state["original_query"] == "test query"
    trace = add_trace_entry(state, "plan", "created plan", iteration=1)
    assert trace[0]["node"] == "plan"
