"""Robots.txt compliance checker with per-domain caching."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)


class RobotsChecker:
    """Caches RobotFileParser instances per domain and checks fetch permissions."""

    def __init__(self, http_client: httpx.AsyncClient, user_agent: str) -> None:
        self._http_client = http_client
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._lock = asyncio.Lock()

    async def _get_parser(self, domain: str, scheme: str) -> RobotFileParser:
        """Fetch and cache robots.txt for a domain."""
        async with self._lock:
            if domain in self._parsers:
                return self._parsers[domain]

        robots_url = f"{scheme}://{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            headers = {"User-Agent": self._user_agent}
            response = await self._http_client.get(robots_url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser.allow_all = True
        except httpx.HTTPError as exc:
            logger.warning("Could not fetch robots.txt for %s: %s", domain, exc)
            parser.allow_all = True

        async with self._lock:
            self._parsers[domain] = parser
        return parser

    async def can_fetch(self, url: str, user_agent: str | None = None) -> bool:
        """Return True if robots.txt permits fetching the URL."""
        parsed = urlparse(url)
        if not parsed.netloc:
            return False

        agent = user_agent or self._user_agent
        domain = parsed.netloc.lower()
        scheme = parsed.scheme or "https"

        try:
            parser = await self._get_parser(domain, scheme)
            return parser.can_fetch(agent, url)
        except Exception as exc:
            logger.warning("Robots check failed for %s: %s", url, exc)
            return True