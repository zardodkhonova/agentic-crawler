"""Tests for deduplication."""

from __future__ import annotations

from crawler.dedup import DuplicateDetector


def test_dedup_url_and_content() -> None:
    dedup = DuplicateDetector()
    assert dedup.is_new_url("https://example.com/a")
    dedup.mark_seen("https://example.com/a", "hello world")
    assert not dedup.is_new_url("https://example.com/a")
    assert not dedup.is_new_content("hello   world")
