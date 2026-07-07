"""Benchmark evaluation metrics."""

from __future__ import annotations

import re
import string
from typing import Any

# Inline chunk citations produced by the RAG engine (UUID in square brackets).
_CITATION_RE = re.compile(r"\[[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\]", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_LEADING_NUMBERING_RE = re.compile(r"^\s*\d+[\.\)]\s+")


def _strip_citations(text: str) -> str:
    """Remove RAG chunk-id citation markers before tokenization."""
    return _CITATION_RE.sub(" ", text)


def _normalize(text: str) -> str:
    text = _strip_citations(text)
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def extract_short_answer(prediction: str) -> str:
    """Extract the direct short answer for scoring purposes.

    The RAG prompt instructs the model to put the shortest direct answer
    on the first line, with an optional clarifying sentence on a second
    line. TriviaQA gold answers are short phrases (e.g. "Paris", "H2O"),
    so scoring against the model's full multi-line response -- which may
    legitimately include a correct explanatory sentence after the direct
    answer -- unfairly penalizes precision/EM even when the model's
    behavior is correct per the prompt's own instructions.

    Some models also prefix a single short answer with list-style
    numbering (e.g. "1. Canada") even when there is only one item, which
    is not meaningful content and should not count as part of the answer
    for scoring purposes -- it is stripped here for the same reason
    citation markers are stripped.

    This only changes what is scored, not what is stored, displayed, or
    validated -- the full answer_text (with citations and any clarifying
    sentence) is still used everywhere else, including hallucination
    checking, so grounding/refusal behavior is unaffected.
    """
    if not prediction:
        return prediction
    first_line = prediction.strip().split("\n", 1)[0].strip()
    first_line = _LEADING_NUMBERING_RE.sub("", first_line).strip()
    return first_line if first_line else prediction


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    """Return True when prediction matches any normalized gold answer."""
    pred = _normalize(extract_short_answer(prediction))
    return any(pred == _normalize(answer) for answer in gold_answers)


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Return max token-level F1 against any gold answer."""
    pred_tokens = _normalize(extract_short_answer(prediction)).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize(gold).split()
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate EM, F1, confidence, and crawl statistics."""
    if not results:
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "avg_confidence": 0.0,
            "hallucination_rate": 0.0,
            "avg_pages_crawled": 0.0,
            "avg_depth": 0.0,
            "avg_latency_seconds": 0.0,
        }

    count = len(results)
    return {
        "exact_match": sum(1 for r in results if r.get("exact_match")) / count,
        "f1": sum(float(r.get("f1", 0.0)) for r in results) / count,
        "avg_confidence": sum(float(r.get("confidence", 0.0)) for r in results) / count,
        "hallucination_rate": sum(float(r.get("hallucination_rate", 0.0)) for r in results) / count,
        "avg_pages_crawled": sum(float(r.get("pages_crawled", 0.0)) for r in results) / count,
        "avg_depth": sum(float(r.get("max_depth", 0.0)) for r in results) / count,
        "avg_latency_seconds": sum(float(r.get("latency_seconds", 0.0)) for r in results) / count,
    }