"""Tests for benchmark metrics."""

from __future__ import annotations

from evaluation.metrics import aggregate_metrics, exact_match, f1_score


def test_exact_match_and_f1() -> None:
    assert exact_match("Paris", ["Paris", "paris"])
    assert exact_match("The capital is Paris", ["Paris"]) is False
    assert f1_score("the capital is Paris", ["Paris"]) > 0


def test_citation_stripping_does_not_hurt_f1() -> None:
    cited = "Paris [550e8400-e29b-41d4-a716-446655440000]"
    assert f1_score(cited, ["Paris"]) == f1_score("Paris", ["Paris"])
    assert exact_match("Paris [550e8400-e29b-41d4-a716-446655440000]", ["Paris"])


def test_article_normalization() -> None:
    assert exact_match("the Eiffel Tower", ["Eiffel Tower"])


def test_aggregate_metrics() -> None:
    results = [
        {"exact_match": True, "f1": 1.0, "confidence": 0.9, "hallucination_rate": 0.1,
         "pages_crawled": 3, "max_depth": 1, "latency_seconds": 2.0},
        {"exact_match": False, "f1": 0.5, "confidence": 0.5, "hallucination_rate": 0.5,
         "pages_crawled": 5, "max_depth": 2, "latency_seconds": 4.0},
    ]
    agg = aggregate_metrics(results)
    assert agg["exact_match"] == 0.5
    assert agg["avg_pages_crawled"] == 4.0
