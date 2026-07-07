"""Tests for page fetcher."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx
import pytest

from crawler.fetcher import PageFetcher
from crawler.rate_limiter import DomainRateLimiter
from crawler.robots import RobotsChecker

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.asyncio
async def test_fetch_returns_document() -> None:
    html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if "robots.txt" in str(request.url):
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = PageFetcher(
        client,
        RobotsChecker(client, "AgenticCrawler/1.0"),
        DomainRateLimiter(10.0),
    )
    doc = await fetcher.fetch("https://example.com/python")
    assert doc is not None
    assert "Python" in doc.raw_html
    assert isinstance(doc.fetched_at, datetime)
    await client.aclose()
