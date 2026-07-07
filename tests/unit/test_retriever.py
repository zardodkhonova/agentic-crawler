"""Tests for semantic retriever."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from indexer.models import Chunk
from retrieval.retriever import SemanticRetriever


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.last_k = 0

    def search(self, query_embedding, k: int):
        self.last_k = k
        now = datetime.now(timezone.utc)
        return [
            Chunk("a", "Alpha", "https://a", 0, now, 1.0),
            Chunk("b", "Beta", "https://b", 1, now, 1.0),
            Chunk("c", "Gamma", "https://c", 2, now, 1.0),
        ]


class FakeReranker:
    def rerank(self, query: str, chunks, top_k: int):
        return list(reversed(chunks))[:top_k]


def test_retriever_search() -> None:
    retriever = SemanticRetriever(FakeEmbedder(), FakeStore())
    results = retriever.search("python", k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_retriever_expands_k_when_reranking() -> None:
    store = FakeStore()
    retriever = SemanticRetriever(
        FakeEmbedder(),
        store,
        reranker=FakeReranker(),
        retrieve_multiplier=3,
    )
    results = retriever.search("python", k=2)
    assert store.last_k == 6
    assert len(results) == 2
    assert results[0].chunk_id == "c"
