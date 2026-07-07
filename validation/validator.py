"""Answer validation orchestrator."""

from __future__ import annotations

import logging

from validation.claim_extractor import ClaimExtractor
from validation.entailment_checker import EntailmentChecker
from validation.schemas import ClaimVerdict, SourceChunk, ValidationResult

logger = logging.getLogger(__name__)

_NO_KNOWLEDGE_PREFIXES = ("i don't know", "i do not know")


def _short_answer_line(answer_text: str) -> str:
    """Return just the first line of the answer, matching the RAG prompt's
    'shortest direct answer on line 1' format contract."""
    return answer_text.strip().split("\n", 1)[0].strip().lower()


class ValidationError(Exception):
    """Raised when validation cannot complete due to a fatal error."""


class Validator:
    """Validates answers by decomposing claims and checking entailment."""

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        entailment_checker: EntailmentChecker,
        confidence_threshold: float = 0.75,
    ) -> None:
        self._claim_extractor = claim_extractor
        self._entailment_checker = entailment_checker
        self._confidence_threshold = confidence_threshold

    async def validate(
        self,
        answer_text: str,
        sources: list[SourceChunk],
    ) -> ValidationResult:
        """Decompose claims and compute confidence from entailment checks.

        The "no knowledge" short-circuit only fires when the answer's
        designated short-answer line (line 1, per the RAG prompt's format
        contract) itself is exactly/starts with "I don't know" -- not
        merely because the phrase appears somewhere in a longer,
        hedge-and-then-answer response. This avoids zeroing out confidence
        for answers that legitimately provide a cited fact after an
        initial hedge.
        """
        short_line = _short_answer_line(answer_text)
        if not short_line or short_line.startswith(_NO_KNOWLEDGE_PREFIXES):
            logger.info("Answer indicates no knowledge; forcing confidence to 0.0")
            return ValidationResult(confidence=0.0, verdicts=[], unsupported_claims=[])

        try:
            claims = await self._claim_extractor.decompose(answer_text)
        except Exception as exc:
            logger.error("Claim extraction failed: %s", exc)
            raise ValidationError(f"Claim extraction failed: {exc}") from exc

        if not claims:
            return ValidationResult(confidence=0.0, verdicts=[], unsupported_claims=[])

        verdicts: list[ClaimVerdict] = []
        supported = 0
        distinct_sources: set[str] = set()

        for claim in claims:
            label = "unsupported"
            supporting_ids: list[str] = []
            for source in sources:
                try:
                    nli_label = self._entailment_checker.check(claim.text, source.text)
                except Exception as exc:
                    logger.warning("Skipping entailment for chunk %s: %s", source.chunk_id, exc)
                    continue
                if nli_label == "entailment":
                    label = "supported"
                    supporting_ids.append(source.chunk_id)
                    distinct_sources.add(source.source_url)
                elif nli_label == "contradiction" and label != "supported":
                    label = "contradicted"

            if label == "supported":
                supported += 1
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    label=label,  # type: ignore[arg-type]
                    supporting_chunk_ids=supporting_ids,
                )
            )

        total = len(claims)
        base_confidence = supported / total if total else 1.0
        source_bonus = min(0.05, 0.01 * len(distinct_sources))
        confidence = min(1.0, base_confidence + source_bonus)

        unsupported = [v.claim for v in verdicts if v.label != "supported"]
        return ValidationResult(
            confidence=confidence,
            verdicts=verdicts,
            unsupported_claims=unsupported,
        )