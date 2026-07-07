"""Tests for text chunker."""

from __future__ import annotations

from datetime import datetime, timezone

from indexer.chunker import TextChunker


def test_chunker_splits_long_text() -> None:
    chunker = TextChunker(chunk_size_tokens=20, chunk_overlap_tokens=5)
    text = ". ".join([f"Sentence number {i} has several words" for i in range(20)])
    chunks = chunker.chunk(text, "https://example.com", datetime.now(timezone.utc), 1.0)
    assert len(chunks) >= 2
    assert all(chunk.chunk_id for chunk in chunks)
