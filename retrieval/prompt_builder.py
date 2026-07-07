"""RAG prompt construction."""

from __future__ import annotations

from retrieval.models import Chunk


def build_rag_prompt(query: str, chunks: list[Chunk]) -> str:
    """Build a prompt with sequential [1], [2], [3]... citation markers.

    Sequential integer citations are used instead of raw chunk-ID UUIDs
    because small local LLMs are unreliable at reproducing long random
    hex strings exactly, which caused citation parsing to fail. The
    mapping from sequential number back to the real chunk UUID is handled
    by the caller (RagEngine.answer), so nothing downstream of the LLM
    call needs to know sequential citations were ever used.
    """
    context_blocks: list[str] = []
    for idx, chunk in enumerate(chunks):
        context_blocks.append(
            f"[{idx + 1}] (source: {chunk.source_url})\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No context available."

    return f"""Answer the question using ONLY the provided context chunks.

Question: {query}

Context:
{context}

Answer format (follow exactly):
1. First line: the shortest direct answer possible (a name, place, date, number, or formula).
   Put citations on this line using the bracketed numbers shown above, e.g. "Paris [1]" or "H2O [2]".
2. Optional second line: one brief clarifying sentence ONLY if the first line needs context.
3. Do NOT write long paragraphs, hedging, or preamble ("Based on the context...").

Rules:
- If the context is insufficient to answer, respond with exactly: I don't know
- Every factual statement must include at least one bracketed citation number, e.g. [1].
- Do not invent facts not supported by the context.
- Use ONLY the bracketed numbers shown in the context above (e.g. [1], [2]) as citations.
  Do not invent citation numbers that were not shown.
"""