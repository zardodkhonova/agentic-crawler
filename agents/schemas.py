"""Agent-level data schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class Plan:
    """LLM-generated crawl and retrieval plan."""

    sub_queries: list[str]
    seed_search_terms: list[str]
    stop_condition_description: str


@dataclass
class CrawlAction:
    """A prioritized URL to crawl."""

    url: str
    depth: int
    source_sub_query: str
    priority_score: float


@dataclass
class FetchedDocument:
    """Raw fetched page content."""

    url: str
    raw_html: str
    fetched_at: datetime
    http_status: int


@dataclass
class Chunk:
    """Indexed text chunk."""

    chunk_id: str
    text: str
    source_url: str
    chunk_index: int
    crawled_at: datetime
    relevance_score_at_crawl: float


@dataclass
class Claim:
    """Atomic claim extracted from an answer."""

    claim_id: str
    text: str


@dataclass
class ClaimVerdict:
    """Entailment verdict for a claim."""

    claim: Claim
    label: Literal["supported", "unsupported", "contradicted"]
    supporting_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Validation outcome."""

    confidence: float
    verdicts: list[ClaimVerdict]
    unsupported_claims: list[Claim]


@dataclass
class AnswerResult:
    """Final agent output."""

    answer_text: str
    sources: list[Chunk]
    confidence: float
    iterations_used: int
    pages_crawled: int
    trace: list[dict[str, Any]] = field(default_factory=list)
