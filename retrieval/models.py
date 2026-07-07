"""Retrieval-local chunk model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Chunk:
    """Text chunk used in retrieval prompts."""

    chunk_id: str
    text: str
    source_url: str
    chunk_index: int
    crawled_at: datetime
    relevance_score_at_crawl: float
