"""Tests for RAG prompt builder."""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.models import Chunk
from retrieval.prompt_builder import build_rag_prompt


def test_build_rag_prompt_includes_chunk_ids() -> None:
    chunks = [
        Chunk(
            chunk_id="id-1",
            text="Paris is the capital of France.",
            source_url="https://example.com",
            chunk_index=0,
            crawled_at=datetime.now(timezone.utc),
            relevance_score_at_crawl=1.0,
        )
    ]
    prompt = build_rag_prompt("Capital of France?", chunks)
    # Chunks are shown to the LLM with sequential numeric markers ([1], [2], ...)
    # rather than raw chunk_id UUIDs, since small local LLMs are unreliable at
    # reproducing long random hex strings exactly. The real chunk_id is mapped
    # back internally by RagEngine after the LLM responds, so it should not
    # appear directly in the prompt text itself.
    assert "[1]" in prompt
    assert "id-1" not in prompt
    assert "Paris" in prompt