"""Tests for benchmark loader."""

from __future__ import annotations

from evaluation.benchmark_loader import load_triviaqa_subset


def test_load_triviaqa_subset_fallback() -> None:
    samples = load_triviaqa_subset(2, seed=7)
    assert len(samples) <= 2
    assert "question" in samples[0]
    assert "answers" in samples[0]
