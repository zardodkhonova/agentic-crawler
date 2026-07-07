"""Sentence-aware text chunking."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime

from indexer.models import Chunk

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")


class TextChunker:
    """Splits text into overlapping token-budget chunks."""

    def __init__(self, chunk_size_tokens: int = 300, chunk_overlap_tokens: int = 50) -> None:
        self._chunk_size = max(chunk_size_tokens, 50)
        self._overlap = min(chunk_overlap_tokens, chunk_size_tokens // 2)

    def chunk(
        self,
        text: str,
        source_url: str,
        crawled_at: datetime,
        relevance_score: float,
    ) -> list[Chunk]:
        """Split text into Chunk objects with UUID identifiers."""
        if not text.strip():
            return []

        sentences = _SENTENCE_RE.split(text.strip())
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        chunk_index = 0

        def flush() -> None:
            nonlocal chunk_index, current, current_tokens
            if not current:
                return
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=chunk_text,
                        source_url=source_url,
                        chunk_index=chunk_index,
                        crawled_at=crawled_at,
                        relevance_score_at_crawl=relevance_score,
                    )
                )
                chunk_index += 1
            overlap: list[str] = []
            overlap_tokens = 0
            for sentence in reversed(current):
                tokens = self._estimate_tokens(sentence)
                if overlap_tokens + tokens > self._overlap:
                    break
                overlap.insert(0, sentence)
                overlap_tokens += tokens
            current = overlap
            current_tokens = overlap_tokens

        for sentence in sentences:
            for part in self._split_if_oversized(sentence.strip()):
                if not part:
                    continue
                tokens = self._estimate_tokens(part)
                if current_tokens + tokens > self._chunk_size and current:
                    flush()
                current.append(part)
                current_tokens += tokens

        flush()
        return chunks

    def _split_if_oversized(self, sentence: str) -> list[str]:
        """Break very long sentences so facts are not buried in huge chunks."""
        if not sentence:
            return []
        if self._estimate_tokens(sentence) <= self._chunk_size:
            return [sentence]

        parts: list[str] = []
        for clause in _CLAUSE_RE.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if self._estimate_tokens(clause) <= self._chunk_size:
                parts.append(clause)
                continue
            words = clause.split()
            step = max(self._chunk_size // 2, 20)
            for start in range(0, len(words), step):
                segment = " ".join(words[start : start + self._chunk_size])
                if segment:
                    parts.append(segment)
        return parts or [sentence]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count as word count."""
        return max(1, len(text.split()))
