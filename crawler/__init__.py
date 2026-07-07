"""Autonomous web crawling agent."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup

from crawler.dedup import DuplicateDetector
from crawler.fetcher import PageFetcher
from crawler.html_extractor import HtmlExtractor
from crawler.link_scorer import LinkRelevanceScorer
from crawler.models import CrawlAction, FetchedDocument
from crawler.priority_queue import CrawlFrontier
from crawler.robots import RobotsChecker
from crawler.url_utils import normalize_url

logger = logging.getLogger(__name__)

_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_WIKIPEDIA_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"

_JUNK_MARKERS = (
    "does not exist",
    "you can create a draft and submit",
    "jump to content search results",
    "wikipedia does not have an article",
)


def _looks_like_junk(text: str) -> bool:
    """Detect low-quality search/placeholder pages that should not be indexed."""
    lowered = text.lower()
    return any(marker in lowered for marker in _JUNK_MARKERS)


class CrawlingAgent:
    """Plans and executes prioritized web crawling steps."""

    def __init__(
        self,
        fetcher: PageFetcher,
        extractor: HtmlExtractor,
        scorer: LinkRelevanceScorer,
        dedup: DuplicateDetector,
        frontier: CrawlFrontier,
        robots_checker: RobotsChecker,
        http_client: httpx.AsyncClient,
        max_pages: int = 25,
        max_depth: int = 3,
    ) -> None:
        self._fetcher = fetcher
        self._extractor = extractor
        self._scorer = scorer
        self._dedup = dedup
        self._frontier = frontier
        self._robots_checker = robots_checker
        self._http_client = http_client
        self._max_pages = max_pages
        self._max_depth = max_depth
        self._pages_crawled = 0
        self._max_depth_reached = 0
        self._sub_queries: list[str] = []

    def reset(self) -> None:
        """Clear all per-query state so the crawler starts fresh for a new question.

        Must be called between evaluation questions (or any new query in
        general) to prevent the frontier, dedup tracker, and page counters
        from one question leaking into the next.
        """
        self._frontier = CrawlFrontier()
        self._dedup = DuplicateDetector()
        self._pages_crawled = 0
        self._max_depth_reached = 0
        self._sub_queries = []

    async def seed(self, plan) -> None:
        """Seed the crawl frontier from plan seed search terms."""
        self._sub_queries = list(getattr(plan, "sub_queries", []) or [])
        seed_terms = list(getattr(plan, "seed_search_terms", []) or [])
        if not seed_terms and self._sub_queries:
            seed_terms = self._sub_queries

        for term in seed_terms:
            urls = await self._search_urls(term)
            for url in urls:
                await self._enqueue(url, depth=0, sub_query=term, priority=1.0)

    async def step(self) -> FetchedDocument | None:
        """Pop the highest-priority URL, fetch it, and expand the frontier."""
        if self._pages_crawled >= self._max_pages:
            logger.info("Max pages reached (%s)", self._max_pages)
            return None

        action = self._frontier.pop()
        if action is None:
            return None

        if not self._dedup.is_new_url(action.url):
            return await self.step()

        document = await self._fetcher.fetch(action.url)
        if document is None:
            self._dedup.mark_seen(action.url)
            return await self.step()

        clean_text, _metadata = self._extractor.extract(document.raw_html, document.url)

        if _looks_like_junk(clean_text):
            logger.info("Skipping junk/placeholder page: %s", document.url)
            self._dedup.mark_seen(document.url, clean_text)
            return await self.step()

        if not self._dedup.is_new_content(clean_text):
            self._dedup.mark_seen(document.url, clean_text)
            return await self.step()

        self._dedup.mark_seen(document.url, clean_text)
        self._pages_crawled += 1
        self._max_depth_reached = max(self._max_depth_reached, action.depth)

        if action.depth < self._max_depth:
            await self._expand_frontier(document, action)

        return document

    async def _expand_frontier(
        self,
        document: FetchedDocument,
        action: CrawlAction,
    ) -> None:
        clean_text, _ = self._extractor.extract(document.raw_html, document.url)
        context_snippet = clean_text[:500]
        links = self._extractor.extract_links(document.raw_html, document.url)

        for link_url, anchor_text in links:
            normalized = normalize_url(link_url)
            if not normalized or not self._dedup.is_new_url(normalized):
                continue
            try:
                allowed = await self._robots_checker.can_fetch(normalized)
            except Exception as exc:
                logger.warning("Robots check failed during expansion: %s", exc)
                allowed = True
            if not allowed:
                continue

            sub_query = action.source_sub_query or (self._sub_queries[0] if self._sub_queries else "")
            score = self._scorer.score(
                anchor_text,
                context_snippet,
                sub_query,
                action.depth + 1,
            )
            if score <= 0.01:
                continue
            await self._enqueue(
                normalized,
                depth=action.depth + 1,
                sub_query=sub_query,
                priority=score,
            )

    async def _enqueue(
        self,
        url: str,
        depth: int,
        sub_query: str,
        priority: float,
    ) -> None:
        if depth > self._max_depth:
            return
        action = CrawlAction(
            url=url,
            depth=depth,
            source_sub_query=sub_query,
            priority_score=priority,
        )
        self._frontier.push(action)

    async def _search_urls(self, query: str, limit: int = 5) -> list[str]:
        """Discover seed URLs, prioritizing Wikipedia over general web search.

        Wikipedia's full-text search is queried first and its results are
        always included, since it reliably covers the general-knowledge
        factual topics this system targets. DuckDuckGo is used only to
        fill any remaining slots after Wikipedia results are added, rather
        than being allowed to fill the entire quota with noisier general
        web results before Wikipedia gets a chance to contribute at all.
        """
        urls: list[str] = []
        seen: set[str] = set()

        def _add(url: str) -> bool:
            normalized = normalize_url(url)
            if normalized and normalized not in seen:
                domain = urlparse(normalized).netloc
                if domain and "duckduckgo.com" not in domain:
                    seen.add(normalized)
                    urls.append(normalized)
                    return True
            return False

        wiki_urls = await self._wikipedia_opensearch(query, limit)
        for url in wiki_urls:
            _add(url)
            if len(urls) >= limit:
                return urls[:limit]

        try:
            response = await self._http_client.post(
                _DDG_HTML_URL,
                data={"q": query},
                headers={"User-Agent": self._fetcher._user_agent},
                follow_redirects=True,
                timeout=20.0,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            for link in soup.select("a.result__a"):
                href = link.get("href", "")
                if not href:
                    continue
                if "uddg=" in href:
                    from urllib.parse import parse_qs, urlparse as parse_url

                    parsed = parse_url(href)
                    params = parse_qs(parsed.query)
                    if "uddg" in params:
                        href = params["uddg"][0]
                _add(href)
                if len(urls) >= limit:
                    break
        except httpx.HTTPError as exc:
            logger.warning("Search seeding failed for '%s': %s", query, exc)
        except Exception as exc:
            logger.warning("Unexpected search error for '%s': %s", query, exc)

        return urls[:limit]

    async def _wikipedia_opensearch(self, query: str, limit: int = 5) -> list[str]:
        """Fallback seeding via Wikipedia's full-text search API.

        Uses action=query&list=search, which matches on article content
        (not just title prefixes), so natural-language questions can
        still find relevant articles. Returns real article URLs, or an
        empty list when there is genuinely no relevant article -- never
        a placeholder "page does not exist" search-results page.
        """
        try:
            response = await self._http_client.get(
                _WIKIPEDIA_OPENSEARCH_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                },
                headers={"User-Agent": self._fetcher._user_agent},
                timeout=20.0,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("query", {}).get("search", [])
            urls = []
            for result in results:
                title = result.get("title", "")
                if not title:
                    continue
                article_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                normalized = normalize_url(article_url)
                if normalized:
                    urls.append(normalized)
            return urls
        except httpx.HTTPError as exc:
            logger.warning("Wikipedia search failed for '%s': %s", query, exc)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Wikipedia search response malformed for '%s': %s", query, exc)
        return []

    def stats(self) -> dict[str, int | float]:
        """Return crawl statistics."""
        return {
            "pages_crawled": self._pages_crawled,
            "frontier_size": len(self._frontier),
            "max_depth_reached": self._max_depth_reached,
        }