"""Sentence-transformer embedding wrapper."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps SentenceTransformer for batched normalized embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                logger.error("Failed to load embedding model %s: %s", self._model_name, exc)
                raise
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings for a batch of texts."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        try:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise
