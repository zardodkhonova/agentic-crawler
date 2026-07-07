"""Tests for link relevance scorer."""

from __future__ import annotations

import numpy as np

from crawler.link_scorer import LinkRelevanceScorer


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            value = 1.0 if "python" in text.lower() else 0.2
            vectors.append([value, 0.0])
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


def test_link_scorer_prefers_relevant_anchor() -> None:
    scorer = LinkRelevanceScorer(FakeEmbedder())
    high = scorer.score("Python docs", "programming language", "python language", depth=0)
    low = scorer.score("unrelated", "misc", "python language", depth=2)
    assert high > low
