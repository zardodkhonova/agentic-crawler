"""Validation data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SourceChunk:
    """Minimal chunk representation for validation."""

    chunk_id: str
    text: str
    source_url: str


@dataclass
class Claim:
    """An atomic factual claim extracted from an answer."""

    claim_id: str
    text: str


@dataclass
class ClaimVerdict:
    """Entailment verdict for a single claim."""

    claim: Claim
    label: Literal["supported", "unsupported", "contradicted"]
    supporting_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Aggregate validation outcome."""

    confidence: float
    verdicts: list[ClaimVerdict]
    unsupported_claims: list[Claim]
