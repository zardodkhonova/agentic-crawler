"""Tests for embedder."""

from __future__ import annotations

import numpy as np
import pytest

from indexer.embedder import Embedder


@pytest.mark.slow
def test_embedder_returns_normalized_vectors() -> None:
    embedder = Embedder("all-MiniLM-L6-v2")
    vectors = embedder.embed(["hello world", "goodbye world"])
    assert vectors.shape[0] == 2
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
