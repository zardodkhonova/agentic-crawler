"""RAG answer generation."""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from retrieval.models import Chunk
from retrieval.prompt_builder import build_rag_prompt
from retrieval.retriever import SemanticRetriever

logger = logging.getLogger(__name__)

_SEQ_CITATION_RE = re.compile(r"\[(\d+)\]")


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str: ...


def _merge_dedupe_chunks(chunk_lists: list[list[Chunk]]) -> list[Chunk]:
    """Merge multiple chunk lists, deduplicating by chunk_id and preserving
    the order of first appearance (original query's results first, then
    any additional chunks found only via sub-query retrieval)."""
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunks in chunk_lists:
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)
    return merged


class RagEngine:
    """Retrieves context and generates a cited answer via LLM."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        llm_client: LLMClient,
        top_k: int = 6,
    ) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._top_k = top_k

    async def answer(
        self,
        query: str,
        sub_queries: list[str] | None = None,
    ) -> tuple[str, list[Chunk], list[Chunk]]:
        """Generate an answer and return (answer_text, cited_chunks, all_retrieved_chunks).

        If sub_queries are provided (e.g. from a compound question split by
        the Query Planner), retrieval is performed for the original query
        AND each sub-query, then merged and deduplicated by chunk_id. This
        improves recall for compound questions where a single-query search
        might otherwise surface chunks for only one of several distinct
        facts being asked about. If no sub_queries are given, retrieval
        falls back to the original query alone.

        all_retrieved_chunks (every chunk shown to the LLM in the prompt,
        not just the ones it happened to cite) is returned so downstream
        validation can check claims against the full retrieved context,
        rather than being unfairly restricted to only the specific chunks
        the LLM remembered to cite.
        """
        chunk_lists = [self._retriever.search(query, self._top_k)]
        if sub_queries:
            for sub_query in sub_queries:
                if sub_query and sub_query.strip() and sub_query.strip() != query.strip():
                    chunk_lists.append(self._retriever.search(sub_query, self._top_k))

        all_chunks = _merge_dedupe_chunks(chunk_lists)[: self._top_k * max(1, len(chunk_lists))]
        # Keep the final context to a reasonable size even after merging multiple queries.
        all_chunks = all_chunks[: max(self._top_k, min(len(all_chunks), self._top_k + 4))]

        prompt = build_rag_prompt(query, all_chunks)
        system = (
            "You are a factual research assistant. "
            "Answer only from provided context. "
            "Lead with the shortest correct answer on the first line, "
            "then cite the bracketed context numbers shown to you (e.g. [1]). "
            'If evidence is insufficient, respond with exactly: I don\'t know'
        )
        try:
            raw_answer = await self._llm_client.complete(prompt, system=system)
        except Exception as exc:
            logger.error("RAG answer generation failed: %s", exc)
            raw_answer = "I don't know."

        answer_text, cited_chunks = self._resolve_citations(raw_answer, all_chunks)
        if not cited_chunks and all_chunks:
            cited_chunks = all_chunks[:3]

        return answer_text, cited_chunks, all_chunks

    @staticmethod
    def _resolve_citations(
        raw_answer: str,
        chunks: list[Chunk],
    ) -> tuple[str, list[Chunk]]:
        """Replace sequential [n] citations in the LLM's raw output with the
        real chunk_id UUIDs, so every downstream consumer of answer_text
        (citation stripping in validation/claim_extractor.py and
        evaluation/metrics.py, source display, etc.) continues to work
        exactly as before, without needing to know sequential citations
        were used internally for prompting."""
        cited_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        def _replace(match: re.Match) -> str:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    cited_chunks.append(chunk)
                return f"[{chunk.chunk_id}]"
            return match.group(0)

        resolved_text = _SEQ_CITATION_RE.sub(_replace, raw_answer)
        return resolved_text, cited_chunks