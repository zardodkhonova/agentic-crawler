"""Command-line interface and dependency wiring."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import httpx

from agents.agent_graph import AgenticCrawler, build_agent_graph
from agents.planner import QueryPlanner
from config.settings import get_settings
from crawler import CrawlingAgent
from crawler.dedup import DuplicateDetector
from crawler.fetcher import PageFetcher
from crawler.html_extractor import HtmlExtractor
from crawler.link_scorer import LinkRelevanceScorer
from crawler.priority_queue import CrawlFrontier
from crawler.rate_limiter import DomainRateLimiter
from crawler.robots import RobotsChecker
from evaluation.run_benchmark import run_evaluation
from indexer.chunker import TextChunker
from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from indexer import KnowledgeIndexer
from llm.ollama_client import OllamaClient
from logging_setup.logger import configure_logging
from retrieval.rag_engine import RagEngine
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import SemanticRetriever
from validation.claim_extractor import ClaimExtractor
from validation.entailment_checker import EntailmentChecker
from validation.validator import Validator

logger = logging.getLogger(__name__)


def _build_agent(settings):
    """Construct all dependencies and return an AgenticCrawler."""
    http_client = httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        follow_redirects=True,
    )
    rate_limiter = DomainRateLimiter(settings.requests_per_second_per_domain)
    robots_checker = RobotsChecker(http_client, settings.user_agent)
    fetcher = PageFetcher(
        http_client=http_client,
        robots_checker=robots_checker,
        rate_limiter=rate_limiter,
        user_agent=settings.user_agent,
        max_retries=settings.max_retries,
        retry_base_delay=settings.retry_base_delay_seconds,
        min_js_content_length=settings.min_content_length,
    )
    embedder = Embedder(settings.embedding_model)
    dedup = DuplicateDetector()
    frontier = CrawlFrontier()
    crawler = CrawlingAgent(
        fetcher=fetcher,
        extractor=HtmlExtractor(),
        scorer=LinkRelevanceScorer(embedder),
        dedup=dedup,
        frontier=frontier,
        robots_checker=robots_checker,
        http_client=http_client,
        max_pages=settings.max_pages_per_query,
        max_depth=settings.max_crawl_depth,
    )
    vector_store = VectorStore(settings.chroma_path)
    chunker = TextChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    indexer = KnowledgeIndexer(
        cleaner_min_length=settings.min_content_length,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    llm_client = OllamaClient(
        model=settings.ollama_model,
        http_client=http_client,
        base_url=settings.ollama_base_url,
    )
    planner = QueryPlanner(llm_client)
    reranker = None
    if settings.rerank_enabled:
        reranker = CrossEncoderReranker(settings.rerank_model)
    retriever = SemanticRetriever(
        embedder,
        vector_store,
        reranker=reranker,
        retrieve_multiplier=settings.retrieve_multiplier,
    )
    rag_engine = RagEngine(retriever, llm_client, top_k=settings.top_k)
    claim_extractor = ClaimExtractor(llm_client)
    entailment_checker = EntailmentChecker(settings.nli_model)
    validator = Validator(
        claim_extractor,
        entailment_checker,
        confidence_threshold=settings.confidence_threshold,
    )
    graph = build_agent_graph(
        planner=planner,
        crawler=crawler,
        indexer=indexer,
        retriever=retriever,
        validator=validator,
        rag_engine=rag_engine,
        settings=settings,
    )
    return AgenticCrawler(graph, settings, crawler, vector_store=vector_store), http_client


async def _run_query(query: str) -> None:
    settings = get_settings()
    configure_logging(settings)
    agent, http_client = _build_agent(settings)
    try:
        result = await agent.run(query)
        output = {
            "query": query,
            "answer": result.answer_text,
            "confidence": result.confidence,
            "iterations_used": result.iterations_used,
            "pages_crawled": result.pages_crawled,
            "sources": [
                {"chunk_id": s.chunk_id, "url": s.source_url, "text": s.text[:200]}
                for s in result.sources
            ],
            "trace": result.trace,
        }
        print(json.dumps(output, indent=2, default=str))
    finally:
        await http_client.aclose()


async def _run_eval(n_samples: int, seed: int, output_path: str) -> None:
    settings = get_settings()
    configure_logging(settings)
    agent, http_client = _build_agent(settings)
    try:
        results = await run_evaluation(agent, n_samples, seed, output_path)
        print(json.dumps(results["aggregate"], indent=2))
    finally:
        await http_client.aclose()


def main() -> None:
    """Parse CLI arguments and run query or evaluation mode."""
    parser = argparse.ArgumentParser(description="Agentic Crawler for Information Retrieval")
    parser.add_argument("--query", type=str, help="Run a single information retrieval query")
    parser.add_argument("--evaluate", action="store_true", help="Run TriviaQA benchmark evaluation")
    parser.add_argument("--n-samples", type=int, default=5, help="Benchmark sample size")
    parser.add_argument("--seed", type=int, default=42, help="Benchmark random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/benchmark_results.json",
        help="Benchmark output JSON path",
    )
    args = parser.parse_args()

    if args.evaluate:
        asyncio.run(_run_eval(args.n_samples, args.seed, args.output))
    elif args.query:
        asyncio.run(_run_query(args.query))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()