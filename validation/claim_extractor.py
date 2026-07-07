"""LLM-based claim decomposition."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from validation.schemas import Claim

logger = logging.getLogger(__name__)

_SHORT_ANSWER_WORD_LIMIT = 6
_CITATION_RE = re.compile(
    r"\[[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\]",
    re.IGNORECASE,
)


def _strip_citations(text: str) -> str:
    """Remove chunk-id citation markers before using text as claim content.

    Citation brackets are our own generated identifiers, not real-world
    text -- passing them into the claim used for NLI entailment checking
    guarantees no source chunk will ever "entail" the claim, since the
    NLI model has never seen that UUID anywhere in real text. Citations
    must be stripped before the claim is checked against evidence.
    """
    return _CITATION_RE.sub("", text).strip()


@runtime_checkable
class LLMClient(Protocol):
    """Local protocol to avoid cross-package imports."""

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str: ...


class _ClaimsPayload(BaseModel):
    claims: list[str] = Field(default_factory=list)


class ClaimExtractor:
    """Decomposes an answer into atomic verifiable claims.

    Short direct answers (per the RAG prompt's "shortest direct answer on
    line 1" format) are treated as a single claim rather than being sent
    through LLM-based decomposition, which can inconsistently over-split a
    short factual phrase into multiple sub-claims and needlessly lower
    confidence even when the answer is entirely correct. Citation markers
    are always stripped from claim text before it is used for entailment
    checking.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        short_answer_word_limit: int = _SHORT_ANSWER_WORD_LIMIT,
    ) -> None:
        self._llm_client = llm_client
        self._short_answer_word_limit = short_answer_word_limit

    async def decompose(self, answer_text: str) -> list[Claim]:
        """Split answer text into a list of atomic claims."""
        stripped = answer_text.strip()
        if not stripped or stripped.lower() == "i don't know.":
            return []

        first_line_raw = stripped.split("\n", 1)[0].strip()
        first_line_clean = _strip_citations(first_line_raw)

        if len(first_line_clean.split()) <= self._short_answer_word_limit:
            if not first_line_clean:
                return []
            return [Claim(claim_id=str(uuid.uuid4()), text=first_line_clean)]

        system = (
            "You extract atomic factual claims from an answer. "
            "Return JSON with a 'claims' array of short strings."
        )
        prompt = (
            "Extract each distinct factual claim from the answer below.\n"
            f"Answer:\n{answer_text}\n"
            'Return JSON: {"claims": ["claim1", "claim2"]}'
        )
        try:
            raw = await self._llm_client.complete(prompt, system=system, json_mode=True)
            payload = _ClaimsPayload.model_validate(json.loads(raw))
        except (json.JSONDecodeError, PydanticValidationError, TypeError) as exc:
            logger.warning("Claim decomposition parse failed: %s", exc)
            return self._fallback_split(answer_text)
        except Exception as exc:
            logger.warning("Claim decomposition failed: %s", exc)
            return self._fallback_split(answer_text)

        claims: list[Claim] = []
        for text in payload.claims:
            cleaned = _strip_citations(text.strip())
            if cleaned:
                claims.append(Claim(claim_id=str(uuid.uuid4()), text=cleaned))
        return claims or self._fallback_split(answer_text)

    @staticmethod
    def _fallback_split(answer_text: str) -> list[Claim]:
        parts = [_strip_citations(part.strip()) for part in answer_text.split(".") if part.strip()]
        return [
            Claim(claim_id=str(uuid.uuid4()), text=part)
            for part in parts
            if len(part) > 10
        ]