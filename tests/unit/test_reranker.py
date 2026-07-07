"""Tests for cross-encoder reranker."""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.models import Chunk
from retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            if "water" in text.lower():
                scores.append(0.9)
            else:
                scores.append(0.1)
        return scores


def test_reranker_prefers_relevant_chunk(monkeypatch) -> None:
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load_model", lambda: FakeCrossEncoder())

    now = datetime.now(timezone.utc)
    chunks = [
        Chunk("a", "Python programming language", "https://a", 0, now, 1.0),
        Chunk("b", "Water chemical formula H2O", "https://b", 1, now, 1.0),
    ]
    ranked = reranker.rerank("What is the chemical formula of water?", chunks, top_k=1)
    assert ranked[0].chunk_id == "b"
