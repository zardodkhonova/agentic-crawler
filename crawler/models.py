"""Crawler-local data models (mirrors agents.schemas for package isolation)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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
