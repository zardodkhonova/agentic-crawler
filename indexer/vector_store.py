"""Persistent ChromaDB vector store."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from indexer.models import Chunk

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when vector store operations fail irrecoverably."""


class VectorStore:
    """Wraps a persistent ChromaDB collection for chunk storage and search."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "agentic_crawler",
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb

            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except Exception as exc:
            logger.error("ChromaDB initialization failed: %s", exc)
            raise IndexingError(f"ChromaDB initialization failed: {exc}") from exc

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert or update chunks with their embeddings."""
        if not chunks:
            return
        try:
            collection = self._ensure_collection()
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "source_url": chunk.source_url,
                    "chunk_index": chunk.chunk_index,
                    "crawled_at": chunk.crawled_at.isoformat(),
                    "relevance_score_at_crawl": chunk.relevance_score_at_crawl,
                }
                for chunk in chunks
            ]
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )
        except Exception as exc:
            logger.error("Vector store upsert failed: %s", exc)
            raise IndexingError(f"Vector store upsert failed: {exc}") from exc

    def search(self, query_embedding: np.ndarray, k: int) -> list[Chunk]:
        """Return the top-k chunks most similar to the query embedding."""
        try:
            collection = self._ensure_collection()
            if collection.count() == 0:
                return []
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            chunks: list[Chunk] = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            for idx, chunk_id in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                text = documents[idx] if idx < len(documents) else ""
                from datetime import datetime

                crawled_at_raw = meta.get("crawled_at", "")
                try:
                    crawled_at = datetime.fromisoformat(crawled_at_raw)
                except (ValueError, TypeError):
                    crawled_at = datetime.utcnow()
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        source_url=str(meta.get("source_url", "")),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        crawled_at=crawled_at,
                        relevance_score_at_crawl=float(
                            meta.get("relevance_score_at_crawl", 0.0)
                        ),
                    )
                )
            return chunks
        except Exception as exc:
            logger.error("Vector store search failed: %s", exc)
            return []

    def count(self) -> int:
        """Return the number of indexed chunks."""
        try:
            collection = self._ensure_collection()
            return collection.count()
        except Exception as exc:
            logger.error("Vector store count failed: %s", exc)
            return 0

    def reset(self) -> None:
        """Delete and recreate the collection."""
        try:
            if self._client is not None:
                self._client.delete_collection(self._collection_name)
            self._collection = None
            self._ensure_collection()
        except Exception as exc:
            logger.warning("Vector store reset failed: %s", exc)
