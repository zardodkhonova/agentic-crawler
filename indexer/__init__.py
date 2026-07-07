"""Knowledge indexing pipeline."""

from __future__ import annotations

import logging

from indexer.chunker import TextChunker
from indexer.cleaner import clean_text
from indexer.embedder import Embedder
from indexer.models import Chunk
from indexer.vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeIndexer:
    """Cleans, chunks, embeds, and stores crawled documents."""

    def __init__(
        self,
        cleaner_min_length: int,
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._min_length = cleaner_min_length
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def process_and_store(
        self,
        document,
        clean_text_and_metadata: tuple[str, dict[str, str]],
        relevance_score: float,
    ) -> list[Chunk]:
        """Clean, chunk, embed, and upsert a fetched document."""
        text, metadata = clean_text_and_metadata
        cleaned = clean_text(text, min_length=self._min_length)
        if not cleaned:
            title = metadata.get("title", "")
            description = metadata.get("description", "")
            cleaned = clean_text(f"{title}. {description}", min_length=self._min_length)
        if not cleaned:
            logger.info("Skipping short content from %s", getattr(document, "url", ""))
            return []

        crawled_at = getattr(document, "fetched_at", None)
        source_url = getattr(document, "url", metadata.get("source_url", ""))

        try:
            chunks = self._chunker.chunk(
                cleaned,
                source_url=source_url,
                crawled_at=crawled_at,
                relevance_score=relevance_score,
            )
            if not chunks:
                return []
            embeddings = self._embedder.embed([chunk.text for chunk in chunks])
            self._vector_store.upsert(chunks, embeddings)
            return chunks
        except Exception as exc:
            logger.error("Indexing failed for %s: %s", source_url, exc)
            return []
