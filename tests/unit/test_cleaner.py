"""Tests for text cleaner."""

from __future__ import annotations

from indexer.cleaner import clean_text


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_min_length() -> None:
    assert clean_text("short", min_length=200) == ""
