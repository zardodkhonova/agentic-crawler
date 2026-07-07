"""Benchmark runner for the agentic crawler."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from evaluation.benchmark_loader import load_triviaqa_subset
from evaluation.metrics import aggregate_metrics, exact_match, f1_score

logger = logging.getLogger(__name__)


async def run_evaluation(
    agent,
    n_samples: int,
    seed: int,
    output_path: str | Path,
) -> dict:
    """Run the agent on a TriviaQA subset and write results to disk."""
    samples = load_triviaqa_subset(n_samples, seed)
    results: list[dict] = []

    for sample in samples:
        question = str(sample["question"])
        gold_answers = list(sample["answers"])  # type: ignore[arg-type]

        agent.reset()

        start = time.perf_counter()
        try:
            answer = await agent.run(question)
            latency = time.perf_counter() - start
            unsupported = len(getattr(answer, "sources", []))
            total_claims = max(unsupported, 1)
            hallucination_rate = (
                len(getattr(answer, "trace", [])) and
                (1.0 - float(getattr(answer, "confidence", 0.0)))
            ) or 0.0
            if hasattr(answer, "confidence") and answer.confidence < 1.0:
                hallucination_rate = 1.0 - answer.confidence

            row = {
                "question": question,
                "prediction": answer.answer_text,
                "gold_answers": gold_answers,
                "exact_match": exact_match(answer.answer_text, gold_answers),
                "f1": f1_score(answer.answer_text, gold_answers),
                "confidence": answer.confidence,
                "hallucination_rate": hallucination_rate,
                "pages_crawled": answer.pages_crawled,
                "max_depth": 0,
                "latency_seconds": latency,
                "iterations_used": answer.iterations_used,
            }
        except Exception as exc:
            logger.error("Evaluation failed for '%s': %s", question, exc)
            latency = time.perf_counter() - start
            row = {
                "question": question,
                "prediction": "",
                "gold_answers": gold_answers,
                "exact_match": False,
                "f1": 0.0,
                "confidence": 0.0,
                "hallucination_rate": 1.0,
                "pages_crawled": 0,
                "max_depth": 0,
                "latency_seconds": latency,
                "error": str(exc),
            }
        results.append(row)
        logger.info(
            "Evaluated: %s | EM=%s F1=%.2f",
            question[:60],
            row["exact_match"],
            row["f1"],
        )

    aggregate = aggregate_metrics(results)
    output = {"results": results, "aggregate": aggregate}

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
    except OSError as exc:
        logger.error("Failed to write benchmark output: %s", exc)

    return output