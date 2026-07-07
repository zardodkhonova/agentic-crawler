"""Tests for HTML extractor."""

from __future__ import annotations

from pathlib import Path

from crawler.html_extractor import HtmlExtractor

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_text_and_links() -> None:
    html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")
    extractor = HtmlExtractor()
    text, meta = extractor.extract(html, "https://example.com/python")
    links = extractor.extract_links(html, "https://example.com/python")
    assert "Python is a high-level" in text
    assert meta["title"] == "Python Programming Language"
    assert any("example.org" in url for url, _ in links)
