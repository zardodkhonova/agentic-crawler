"""HTTP page fetcher with robots.txt, rate limiting, and retries."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from config.retry import async_retry_on_transient
from crawler.rate_limiter import DomainRateLimiter
from crawler.robots import RobotsChecker

logger = logging.getLogger(__name__)


class FetchError(Exception):
    """Raised when a page fetch fails irrecoverably."""


class PageFetcher:
    """Fetches web pages respecting robots.txt and per-domain rate limits."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        robots_checker: RobotsChecker,
        rate_limiter: DomainRateLimiter,
        user_agent: str = "AgenticCrawler/1.0",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        min_js_content_length: int = 200,
    ) -> None:
        self._http_client = http_client
        self._robots_checker = robots_checker
        self._rate_limiter = rate_limiter
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._min_js_content_length = min_js_content_length
        self._playwright_available: bool | None = None

    async def fetch(self, url: str):
        """Fetch a URL and return a FetchedDocument or None on graceful failure."""
        from crawler.models import FetchedDocument

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain:
            logger.warning("Invalid URL skipped: %s", url)
            return None

        try:
            allowed = await self._robots_checker.can_fetch(url, self._user_agent)
        except Exception as exc:
            logger.warning("Robots check error for %s: %s", url, exc)
            allowed = True

        if not allowed:
            logger.info("Blocked by robots.txt: %s", url)
            return None

        try:
            await self._rate_limiter.acquire(domain)
        except Exception as exc:
            logger.warning("Rate limiter error for %s: %s", domain, exc)

        headers = {"User-Agent": self._user_agent}

        @async_retry_on_transient(self._max_retries, self._retry_base_delay)
        async def _get(target_url: str) -> httpx.Response:
            response = await self._http_client.get(
                target_url,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response

        try:
            response = await _get(url)
            raw_html = response.text
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.info("Non-HTML content skipped: %s (%s)", url, content_type)
                return None

            if len(raw_html.strip()) < self._min_js_content_length:
                js_html = await self._try_playwright_render(url)
                if js_html:
                    raw_html = js_html

            return FetchedDocument(
                url=str(response.url),
                raw_html=raw_html,
                fetched_at=datetime.now(timezone.utc),
                http_status=response.status_code,
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None
        except httpx.HTTPError as exc:
            logger.warning("Network error fetching %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.error("Unexpected fetch error for %s: %s", url, exc)
            return None

    async def _try_playwright_render(self, url: str) -> str | None:
        """Render a page with Playwright when static fetch yields little content."""
        if self._playwright_available is False:
            return None
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                try:
                    page = await browser.new_page(user_agent=self._user_agent)
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    content = await page.content()
                    if len(content.strip()) >= self._min_js_content_length:
                        self._playwright_available = True
                        return content
                finally:
                    await browser.close()
        except ImportError:
            logger.debug("Playwright not installed; skipping JS rendering")
            self._playwright_available = False
        except Exception as exc:
            logger.warning("Playwright render failed for %s: %s", url, exc)
            self._playwright_available = False
        return None
