"""Semantic retrieval over the vector store."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from retrieval.models import Chunk

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    def search(self, query_embedding: np.ndarray, k: int) -> list: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]: ...


class SemanticRetriever:
    """Embeds queries and searches the vector store."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        reranker: RerankerProtocol | None = None,
        retrieve_multiplier: int = 3,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._retrieve_multiplier = max(retrieve_multiplier, 1)

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return top-k chunks for a natural-language query."""
        try:
            embedding = self._embedder.embed([query])[0]
            fetch_k = k
            if self._reranker is not None:
                fetch_k = k * self._retrieve_multiplier
            raw_chunks = self._vector_store.search(embedding, fetch_k)
            chunks = [_to_retrieval_chunk(chunk) for chunk in raw_chunks]
            if self._reranker is not None and len(chunks) > k:
                chunks = self._reranker.rerank(query, chunks, k)
            return chunks[:k]
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", query, exc)
            return []


def _to_retrieval_chunk(chunk) -> Chunk:
    return Chunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_url=chunk.source_url,
        chunk_index=chunk.chunk_index,
        crawled_at=chunk.crawled_at,
        relevance_score_at_crawl=chunk.relevance_score_at_crawl,
    )
