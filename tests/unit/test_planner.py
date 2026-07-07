"""Tests for query planner."""

from __future__ import annotations

import json

import pytest

from agents.planner import QueryPlanner


class FakeLLM:
    async def complete(self, prompt, system=None, json_mode=False) -> str:
        return json.dumps(
            {
                "sub_queries": ["Who created Python?"],
                "seed_search_terms": ["Python creator"],
                "stop_condition_description": "Found creator info",
            }
        )


@pytest.mark.asyncio
async def test_query_planner() -> None:
    planner = QueryPlanner(FakeLLM())
    plan = await planner.plan("Who created Python?")
    assert plan.sub_queries
    assert plan.seed_search_terms
