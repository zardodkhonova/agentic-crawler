"""URL and content deduplication."""

from __future__ import annotations

import hashlib


class DuplicateDetector:
    """Tracks seen URLs and content hashes to avoid duplicate crawling."""

    def __init__(self) -> None:
        self._seen_urls: set[str] = set()
        self._seen_content_hashes: set[str] = set()

    def is_new_url(self, url: str) -> bool:
        """Return True if the URL has not been seen."""
        return url not in self._seen_urls

    def is_new_content(self, normalized_text: str) -> bool:
        """Return True if normalized text content has not been seen."""
        content_hash = self._hash_text(normalized_text)
        return content_hash not in self._seen_content_hashes

    def mark_seen(self, url: str, normalized_text: str | None = None) -> None:
        """Record a URL and optional content hash as seen."""
        self._seen_urls.add(url)
        if normalized_text:
            self._seen_content_hashes.add(self._hash_text(normalized_text))

    @staticmethod
    def _hash_text(text: str) -> str:
        normalized = " ".join(text.split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
