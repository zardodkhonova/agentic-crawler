"""Tests for RAG engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from retrieval.models import Chunk
from retrieval.rag_engine import RagEngine
from retrieval.retriever import SemanticRetriever


class FakeEmbedder:
    def embed(self, texts):
        import numpy as np

        return np.ones((len(texts), 3), dtype=np.float32)


class FakeStore:
    def search(self, query_embedding, k: int):
        return [
            Chunk(
                chunk_id="11111111-1111-1111-1111-111111111111",
                text="Guido van Rossum created Python.",
                source_url="https://example.com/python",
                chunk_index=0,
                crawled_at=datetime.now(timezone.utc),
                relevance_score_at_crawl=1.0,
            )
        ]


class FakeLLM:
    async def complete(self, prompt, system=None, json_mode=False) -> str:
        # The LLM now sees sequential citation markers ([1], [2], ...) in the
        # prompt instead of raw UUIDs, so the fake LLM's response uses the
        # sequential marker it would actually have been shown for the single
        # retrieved chunk in this test (chunk index 0 -> marker [1]).
        return "Python was created by Guido van Rossum [1]."


@pytest.mark.asyncio
async def test_rag_engine_answer() -> None:
    retriever = SemanticRetriever(FakeEmbedder(), FakeStore())
    engine = RagEngine(retriever, FakeLLM(), top_k=1)
    answer, cited_chunks, all_chunks = await engine.answer("Who created Python?")
    # The returned answer_text should have the sequential [1] marker resolved
    # back to the real chunk_id UUID, so downstream citation-stripping logic
    # (validation/claim_extractor.py, evaluation/metrics.py) continues to work
    # against real UUID citations without needing any changes.
    assert "Guido" in answer
    assert "11111111-1111-1111-1111-111111111111" in answer
    assert cited_chunks
    assert all_chunks
    assert cited_chunks[0].chunk_id == "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_rag_engine_multi_query_merges_chunks() -> None:
    """When sub_queries are provided, retrieval should run for each of them
    in addition to the original query, with results merged and deduplicated
    by chunk_id -- verifying the multi-query retrieval fix for compound
    questions where a single query search might otherwise miss chunks
    relevant to a distinct sub-fact."""
    retriever = SemanticRetriever(FakeEmbedder(), FakeStore())
    engine = RagEngine(retriever, FakeLLM(), top_k=1)
    answer, cited_chunks, all_chunks = await engine.answer(
        "Who created Python and when?",
        sub_queries=["Who created Python?", "When was Python created?"],
    )
    # FakeStore always returns the same single chunk regardless of query text,
    # so merging must correctly deduplicate rather than producing duplicates.
    chunk_ids = [c.chunk_id for c in all_chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all_chunks