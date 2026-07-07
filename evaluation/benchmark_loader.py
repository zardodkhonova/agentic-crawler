"""TriviaQA benchmark loading via HuggingFace datasets."""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def load_triviaqa_subset(n_samples: int, seed: int = 42) -> list[dict[str, list[str] | str]]:
    """Load a deterministic subset of TriviaQA (rc.nocontext)."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    except Exception as exc:
        logger.error(
            "Failed to load TriviaQA dataset from HuggingFace: %s. "
            "Falling back to a small offline sample set (%s questions) -- "
            "these are NOT randomly sampled from the real benchmark and "
            "should not be used for reported evaluation results.",
            exc,
            n_samples,
        )
        return _fallback_samples(n_samples, seed)

    if n_samples > len(dataset):
        logger.warning(
            "Requested %s samples but dataset only has %s rows; using all available.",
            n_samples,
            len(dataset),
        )

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = indices[:n_samples]

    samples: list[dict[str, list[str] | str]] = []
    for idx in selected:
        row = dataset[idx]
        question = row.get("question", "")
        answer_dict = row.get("answer", {})
        aliases = answer_dict.get("aliases", []) if isinstance(answer_dict, dict) else []
        value = answer_dict.get("value", "") if isinstance(answer_dict, dict) else ""
        answers = [value] + list(aliases) if value else list(aliases)
        answers = [a for a in answers if a]
        if question and answers:
            samples.append({"question": question, "answers": answers})

    if len(samples) < n_samples:
        logger.warning(
            "Only %s of %s requested samples had usable question/answer data.",
            len(samples),
            n_samples,
        )

    return samples


def _fallback_samples(n_samples: int, seed: int) -> list[dict[str, list[str] | str]]:
    """Offline fallback when HuggingFace datasets are unavailable.

    WARNING: these are a fixed set of 3 hand-picked easy questions, not a
    random sample of TriviaQA. This function exists only so the CLI does
    not crash when there is no internet access; results produced using
    this fallback must not be reported as TriviaQA benchmark results.
    """
    logger.warning(
        "Using offline fallback questions -- results from this run are "
        "NOT valid TriviaQA benchmark scores and should not be reported."
    )
    rng = random.Random(seed)
    base = [
        {"question": "What is the capital of France?", "answers": ["Paris"]},
        {"question": "Who wrote Romeo and Juliet?", "answers": ["William Shakespeare", "Shakespeare"]},
        {"question": "What is the chemical symbol for water?", "answers": ["H2O"]},
    ]
    rng.shuffle(base)
    return base[:n_samples]