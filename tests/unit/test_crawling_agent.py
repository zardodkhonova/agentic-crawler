"""Tests for crawling agent."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from crawler import CrawlingAgent
from crawler.dedup import DuplicateDetector
from crawler.fetcher import PageFetcher
from crawler.html_extractor import HtmlExtractor
from crawler.link_scorer import LinkRelevanceScorer
from crawler.models import CrawlAction
from crawler.priority_queue import CrawlFrontier
from crawler.rate_limiter import DomainRateLimiter
from crawler.robots import RobotsChecker

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakePlan:
    sub_queries = ["Python programming"]
    seed_search_terms = ["Python programming language"]


class FakeEmbedder:
    def embed(self, texts):
        import numpy as np

        return np.ones((len(texts), 3), dtype=np.float32)


@pytest.mark.asyncio
async def test_crawling_agent_step(monkeypatch) -> None:
    html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")

    async def mock_seed(self, plan) -> None:
        self._frontier.push(
            CrawlAction("https://example.com/python", 0, "Python", 1.0)
        )

    monkeypatch.setattr(CrawlingAgent, "seed", mock_seed)

    def handler(request: httpx.Request) -> httpx.Response:
        if "robots.txt" in str(request.url):
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    robots = RobotsChecker(client, "AgenticCrawler/1.0")
    agent = CrawlingAgent(
        PageFetcher(client, robots, DomainRateLimiter(100.0)),
        HtmlExtractor(),
        LinkRelevanceScorer(FakeEmbedder()),
        DuplicateDetector(),
        CrawlFrontier(),
        robots,
        client,
        max_pages=1,
        max_depth=0,
    )
    await agent.seed(FakePlan())
    doc = await agent.step()
    await client.aclose()
    assert doc is not None
    assert "Python" in doc.raw_html
