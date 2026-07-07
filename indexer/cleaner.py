"""Text cleaning utilities."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw_text: str, min_length: int = 0) -> str:
    """Collapse whitespace and enforce minimum content length."""
    cleaned = _WHITESPACE_RE.sub(" ", raw_text or "").strip()
    if min_length and len(cleaned) < min_length:
        return ""
    return cleaned
