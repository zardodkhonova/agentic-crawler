"""Indexer-local chunk model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Chunk:
    """A text chunk stored in the vector index."""

    chunk_id: str
    text: str
    source_url: str
    chunk_index: int
    crawled_at: datetime
    relevance_score_at_crawl: float
