"""Tests for robots.txt checker."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from crawler.robots import RobotsChecker

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.asyncio
async def test_robots_allows_public_path() -> None:
    robots_text = (FIXTURES / "robots.txt").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("robots.txt"):
            return httpx.Response(200, text=robots_text)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    checker = RobotsChecker(client, "AgenticCrawler/1.0")
    allowed = await checker.can_fetch("https://example.com/about")
    assert allowed is True
    await client.aclose()
