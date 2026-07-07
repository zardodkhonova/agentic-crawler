"""Cross-encoder reranking of retrieved chunks."""

from __future__ import annotations

import logging

from retrieval.models import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved chunks by query-chunk relevance using a cross-encoder.

    Unlike the bi-encoder used for initial vector retrieval (which embeds
    the query and each chunk independently), a cross-encoder scores the
    query and chunk together in a single forward pass, producing a much
    more precise relevance judgment at the cost of being too slow to run
    over an entire corpus -- so it is only applied to the small shortlist
    of candidates already retrieved by the vector store.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
            except Exception as exc:
                logger.error("Failed to load reranker model %s: %s", self._model_name, exc)
                raise
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return the top_k chunks most relevant to the query, per the cross-encoder."""
        if not chunks:
            return []
        try:
            model = self._load_model()
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = model.predict(pairs)
            scored_chunks = sorted(
                zip(chunks, scores),
                key=lambda item: item[1],
                reverse=True,
            )
            return [chunk for chunk, _score in scored_chunks[:top_k]]
        except Exception as exc:
            logger.warning("Reranking failed, falling back to original order: %s", exc)
            return chunks[:top_k]