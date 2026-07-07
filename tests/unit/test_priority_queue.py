"""Tests for crawl frontier priority queue."""

from __future__ import annotations

from crawler.models import CrawlAction
from crawler.priority_queue import CrawlFrontier


def test_frontier_pops_highest_priority() -> None:
    frontier = CrawlFrontier()
    frontier.push(CrawlAction("https://a.com", 0, "q", 0.2))
    frontier.push(CrawlAction("https://b.com", 0, "q", 0.9))
    top = frontier.pop()
    assert top is not None
    assert top.url == "https://b.com"
