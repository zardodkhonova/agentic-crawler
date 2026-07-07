"""Tests for vector store."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from indexer.models import Chunk
from indexer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma")


def test_vector_store_upsert_and_search(store) -> None:
    chunks = [
        Chunk(
            chunk_id="c1",
            text="Python is a programming language",
            source_url="https://example.com",
            chunk_index=0,
            crawled_at=datetime.now(timezone.utc),
            relevance_score_at_crawl=1.0,
        )
    ]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    store.upsert(chunks, embeddings)
    assert store.count() == 1
    results = store.search(embeddings[0], k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
