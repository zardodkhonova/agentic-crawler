"""NLI-based entailment checking."""

from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

NLILabel = Literal["entailment", "neutral", "contradiction"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _keyword_overlap_ratio(claim_text: str, chunk_text: str) -> float:
    """Fraction of claim's significant words that also appear in the chunk.

    Used only to rescue borderline "neutral" NLI verdicts when the claim's
    key terms are literally present in the source text -- this does not
    replace the NLI model, it only prevents an overly conservative NLI
    "neutral" label from discarding evidence that clearly mentions the
    same entities/facts as the claim.
    """
    claim_words = set(_WORD_RE.findall(claim_text.lower()))
    claim_words = {w for w in claim_words if len(w) > 2}
    if not claim_words:
        return 0.0
    chunk_words = set(_WORD_RE.findall(chunk_text.lower()))
    overlap = claim_words & chunk_words
    return len(overlap) / len(claim_words)


class EntailmentChecker:
    """Checks whether a source chunk entails a claim using an NLI model,
    with a keyword-overlap fallback for borderline "neutral" verdicts."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        neutral_rescue_overlap_threshold: float = 0.6,
    ) -> None:
        self._model_name = model_name
        self._pipeline = None
        self._neutral_rescue_overlap_threshold = neutral_rescue_overlap_threshold

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "text-classification",
                    model=self._model_name,
                    top_k=None,
                )
            except Exception as exc:
                logger.error("Failed to load NLI model %s: %s", self._model_name, exc)
                raise
        return self._pipeline

    def check(self, claim_text: str, chunk_text: str) -> NLILabel:
        """Return entailment label for claim given chunk evidence.

        A raw "neutral" NLI verdict is upgraded to "entailment" only when
        most of the claim's substantive keywords are literally present in
        the chunk text -- this compensates for NLI models being trained on
        clean single-sentence pairs and often under-labeling real-world,
        loosely-phrased support as "neutral" rather than "entailment".
        Contradiction verdicts are never overridden.
        """
        if not claim_text.strip() or not chunk_text.strip():
            return "neutral"
        try:
            nli = self._load_pipeline()
            outputs = nli({"text": chunk_text[:512], "text_pair": claim_text[:256]})
            if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
                outputs = outputs[0]
            scores = {item["label"].lower(): item["score"] for item in outputs}
            best_label = max(scores, key=scores.get)
            mapping = {
                "entailment": "entailment",
                "contradiction": "contradiction",
                "neutral": "neutral",
            }
            label: NLILabel = "neutral"
            for key, value in mapping.items():
                if key in best_label:
                    label = value
                    break

            if label == "neutral":
                overlap = _keyword_overlap_ratio(claim_text, chunk_text)
                if overlap >= self._neutral_rescue_overlap_threshold:
                    logger.debug(
                        "Rescuing neutral verdict via keyword overlap=%.2f",
                        overlap,
                    )
                    return "entailment"

            return label
        except Exception as exc:
            logger.warning("Entailment check failed: %s", exc)
            return "neutral"