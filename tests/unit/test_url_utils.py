"""Tests for crawler URL utilities."""

from __future__ import annotations

from crawler.url_utils import normalize_url, same_domain


def test_normalize_url_strips_tracking() -> None:
    url = normalize_url("https://Example.com/page?utm_source=x&id=1#section")
    assert url == "https://example.com/page?id=1"
    assert "#section" not in url


def test_normalize_relative_url() -> None:
    url = normalize_url("/about", "https://example.com/docs/")
    assert url == "https://example.com/about"


def test_same_domain() -> None:
    assert same_domain("https://www.example.com/a", "https://example.com/b")
