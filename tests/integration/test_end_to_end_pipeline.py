"""End-to-end pipeline integration test with mocked HTTP."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from agents.agent_graph import AgenticCrawler, build_agent_graph
from agents.planner import QueryPlanner
from config.settings import Settings
from crawler import CrawlingAgent
from crawler.dedup import DuplicateDetector
from crawler.fetcher import PageFetcher
from crawler.html_extractor import HtmlExtractor
from crawler.link_scorer import LinkRelevanceScorer
from crawler.models import CrawlAction
from crawler.priority_queue import CrawlFrontier
from crawler.rate_limiter import DomainRateLimiter
from crawler.robots import RobotsChecker
from indexer.chunker import TextChunker
from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from indexer import KnowledgeIndexer
from retrieval.rag_engine import RagEngine
from retrieval.retriever import SemanticRetriever
from validation.claim_extractor import ClaimExtractor
from validation.entailment_checker import EntailmentChecker
from validation.validator import Validator

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class MockLLM:
    async def complete(self, prompt, system=None, json_mode=False) -> str:
        if json_mode:
            if "crawl plan" in prompt.lower() or "sub_queries" in prompt:
                return json.dumps(
                    {
                        "sub_queries": ["Who created Python?"],
                        "seed_search_terms": ["Python creator"],
                        "stop_condition_description": "done",
                    }
                )
            if "claims" in prompt.lower():
                return json.dumps({"claims": ["Python was created by Guido van Rossum."]})
        if "chunk_id" in prompt or "Context:" in prompt:
            return (
                "Python was created by Guido van Rossum "
                "[11111111-1111-1111-1111-111111111111]."
            )
        return "Python creator Guido van Rossum"


class MockEmbedder:
    def embed(self, texts):
        import numpy as np

        vectors = []
        for text in texts:
            value = 1.0 if "python" in text.lower() else 0.5
            vectors.append([value, 0.0, 0.0])
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


class MockEntailment:
    def check(self, claim_text, chunk_text) -> str:
        return "entailment"


@pytest.mark.asyncio
async def test_end_to_end_pipeline(tmp_path, monkeypatch) -> None:
    page1 = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")
    page2 = (FIXTURES / "sample_page2.html").read_text(encoding="utf-8")

    async def mock_seed(self, plan) -> None:
        self._frontier.push(
            CrawlAction("https://example.com/python", 0, "Python creator", 1.0)
        )

    monkeypatch.setattr(CrawlingAgent, "seed", mock_seed)

    def handler(request: httpx.Request) -> httpx.Response:
        if "robots.txt" in str(request.url):
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if "python" in str(request.url):
            return httpx.Response(200, text=page1, headers={"content-type": "text/html"})
        return httpx.Response(200, text=page2, headers={"content-type": "text/html"})

    settings = Settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        max_pages_per_query=2,
        max_iterations=2,
        confidence_threshold=0.5,
        top_k=3,
        openrouter_api_key="test",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    robots = RobotsChecker(client, settings.user_agent)
    fetcher = PageFetcher(client, robots, DomainRateLimiter(100.0), settings.user_agent)
    embedder = MockEmbedder()
    frontier = CrawlFrontier()
    crawler = CrawlingAgent(
        fetcher,
        HtmlExtractor(),
        LinkRelevanceScorer(embedder),
        DuplicateDetector(),
        frontier,
        robots,
        client,
        max_pages=2,
        max_depth=1,
    )
    vector_store = VectorStore(tmp_path / "chroma")
    indexer = KnowledgeIndexer(
        cleaner_min_length=10,
        chunker=TextChunker(100, 20),
        embedder=embedder,
        vector_store=vector_store,
    )
    llm = MockLLM()
    planner = QueryPlanner(llm)
    retriever = SemanticRetriever(embedder, vector_store)
    rag_engine = RagEngine(retriever, llm, top_k=settings.top_k)
    validator = Validator(ClaimExtractor(llm), MockEntailment(), settings.confidence_threshold)
    graph = build_agent_graph(
        planner, crawler, indexer, retriever, validator, rag_engine, settings
    )
    agent = AgenticCrawler(graph, settings, crawler)

    result = await agent.run("Who created Python?")
    await client.aclose()

    assert result.answer_text
    assert result.confidence >= 0.0
    assert result.pages_crawled >= 1
    assert isinstance(result.trace, list)
