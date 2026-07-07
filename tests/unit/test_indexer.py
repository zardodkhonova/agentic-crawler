"""Tests for knowledge indexer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from indexer.chunker import TextChunker
from indexer import KnowledgeIndexer


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeVectorStore:
    def __init__(self) -> None:
        self.items: list = []

    def upsert(self, chunks, embeddings) -> None:
        self.items.extend(chunks)


class FakeDocument:
    url = "https://example.com"
    fetched_at = datetime.now(timezone.utc)


def test_process_and_store() -> None:
    indexer = KnowledgeIndexer(
        cleaner_min_length=10,
        chunker=TextChunker(50, 10),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    text = "Python is a widely used programming language. " * 5
    chunks = indexer.process_and_store(FakeDocument(), (text, {"title": "Python"}), 1.0)
    assert chunks
    assert chunks[0].source_url == "https://example.com"
