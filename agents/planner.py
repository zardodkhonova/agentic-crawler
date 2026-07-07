"""LLM-based query planning."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from agents.schemas import Plan

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Local LLM protocol for package isolation."""

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str: ...


class _PlanPayload(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)
    seed_search_terms: list[str] = Field(default_factory=list)
    stop_condition_description: str = ""


def _coerce_string_list(raw_list: Any) -> list[str]:
    """Flatten values that a smaller LLM sometimes returns as dicts/objects
    instead of plain strings (e.g. [{"sub_query_1": "..."}] instead of
    ["..."]), extracting the first string value found in each item.

    This is a general robustness measure for any JSON-mode response shape
    variance from smaller local models -- it does not depend on or assume
    any specific question content.
    """
    coerced: list[str] = []
    if not isinstance(raw_list, list):
        return coerced
    for item in raw_list:
        if isinstance(item, str):
            if item.strip():
                coerced.append(item.strip())
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    coerced.append(value.strip())
                    break
    return coerced


class QueryPlanner:
    """Generates and refines crawl plans from user queries."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def plan(self, query: str) -> Plan:
        """Create an initial crawl plan for a user query."""
        system = (
            "You are a research planner for factual and trivia questions. "
            "Decompose queries into focused sub-queries and concise web search terms. "
            "Return JSON only."
        )
        prompt = (
            f'Create a crawl plan for: "{query}"\n'
            "Return JSON with EXACTLY this shape (both lists must contain plain "
            "strings, never objects or numbered keys):\n"
            '{"sub_queries": ["...", "..."], '
            '"seed_search_terms": ["...", "..."], '
            '"stop_condition_description": "..."}\n\n'
            '- "sub_queries": list of 1-3 focused factual sub-questions, each a plain string\n'
            '- "seed_search_terms": list of 2-5 short keyword phrases (2-6 words each), '
            "each a plain string, suitable for web/Wikipedia search — use entity names, "
            "topics, and article-style phrases, NOT full interrogative sentences\n"
            '- "stop_condition_description": a plain string describing when to stop crawling\n'
            "Guidelines:\n"
            "- Name the subject directly in at least one seed term.\n"
            "- Prefer terms that would match a Wikipedia article title or lead section.\n"
            "- Avoid filler words like 'what is', 'tell me about', 'information on'.\n"
            "- If the question asks about MULTIPLE distinct facts, events, or entities "
            "(e.g. it mentions several unrelated people, dates, or topics joined by 'and'), "
            "create ONE separate sub_query and ONE separate seed_search_term PER DISTINCT "
            "fact — never merge unrelated facts into a single combined search phrase.\n"
            "- Include specific disambiguating details from the question directly in your "
            "seed search terms (exact dates, relationship context, geographic anchors, "
            "numeric qualifiers) — a generic entity name alone is often ambiguous and can "
            "retrieve the wrong specific fact about a broadly related topic. For example, "
            "prefer 'Europe percentage of Earth's land surface area' over just 'Europe', "
            "and prefer 'Jane Asher husband married 1981 children' over just 'Jane Asher'.\n"
            "- Every list item must be a plain string. Do NOT wrap items in objects like "
            '{"sub_query_1": "..."} -- just write the string directly in the list.\n'
        )
        try:
            raw = await self._llm_client.complete(prompt, system=system, json_mode=True)
            data = json.loads(raw)
            if isinstance(data, dict):
                data["sub_queries"] = _coerce_string_list(data.get("sub_queries"))
                data["seed_search_terms"] = _coerce_string_list(data.get("seed_search_terms"))
            payload = _PlanPayload.model_validate(data)
        except (json.JSONDecodeError, PydanticValidationError, TypeError) as exc:
            logger.warning("Plan parse failed, using fallback: %s", exc)
            return Plan(
                sub_queries=[query],
                seed_search_terms=[query],
                stop_condition_description="Enough relevant sources indexed.",
            )
        except Exception as exc:
            logger.error("Planning failed: %s", exc)
            return Plan(
                sub_queries=[query],
                seed_search_terms=[query],
                stop_condition_description="Enough relevant sources indexed.",
            )

        sub_queries = [q.strip() for q in payload.sub_queries if q.strip()] or [query]
        seed_terms = [t.strip() for t in payload.seed_search_terms if t.strip()] or sub_queries
        return Plan(
            sub_queries=sub_queries,
            seed_search_terms=seed_terms,
            stop_condition_description=payload.stop_condition_description
            or "Enough relevant sources indexed.",
        )

    async def refine(self, original_query: str, unsupported_claims: list) -> str:
        """Generate one follow-up sub-query to address unsupported claims."""
        claims_text = "; ".join(
            getattr(c, "text", str(c)) for c in unsupported_claims[:5]
        )
        system = (
            "Return a single short web search query (2-8 keywords) as plain text. "
            "No full sentences or question marks."
        )
        prompt = (
            f"Original question: {original_query}\n"
            f"Unsupported claims: {claims_text}\n"
            "Write one concise keyword search query to find the missing evidence. "
            "Prefer Wikipedia-friendly entity/topic phrases."
        )
        try:
            result = await self._llm_client.complete(prompt, system=system)
            refined = result.strip().strip('"')
            return refined or original_query
        except Exception as exc:
            logger.warning("Plan refinement failed: %s", exc)
            return original_query