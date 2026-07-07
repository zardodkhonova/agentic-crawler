"""Embedding-based link relevance scoring."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEPTH_PENALTY = 0.05


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute true cosine similarity, independent of whether the embedder
    already returns normalized vectors.

    A raw dot product only equals cosine similarity when both vectors have
    unit length; relying on that assumption silently produces wrong scores
    whenever the embedder's output isn't normalized, which can systematically
    misrank crawl links regardless of the query.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class LinkRelevanceScorer:
    """Scores crawl links using cosine similarity minus depth penalty."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def score(
        self,
        anchor_text: str,
        context_snippet: str,
        sub_query: str,
        depth: int,
    ) -> float:
        """Return a relevance score in roughly [0, 1] after depth penalty."""
        link_text = f"{anchor_text.strip()} {context_snippet.strip()}".strip()
        if not link_text or not sub_query.strip():
            return 0.0

        try:
            embeddings = self._embedder.embed([link_text, sub_query])
            if embeddings.shape[0] < 2:
                return 0.0
            similarity = _cosine_similarity(embeddings[0], embeddings[1])
            similarity = max(0.0, min(1.0, similarity))
            penalized = similarity - (_DEPTH_PENALTY * depth)
            return max(0.0, penalized)
        except Exception as exc:
            logger.warning("Link scoring failed: %s", exc)
            return 0.0