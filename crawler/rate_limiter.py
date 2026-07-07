"""Per-domain token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class DomainRateLimiter:
    """Token-bucket rate limiter enforcing requests-per-second per domain."""

    def __init__(self, requests_per_second: float = 1.0) -> None:
        self._rate = max(requests_per_second, 0.1)
        self._interval = 1.0 / self._rate
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        async with self._global_lock:
            if domain not in self._locks:
                self._locks[domain] = asyncio.Lock()
            return self._locks[domain]

    async def acquire(self, domain: str) -> None:
        """Wait until a request to the domain is permitted."""
        lock = await self._get_lock(domain)
        async with lock:
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            wait_time = self._interval - (now - last)
            if wait_time > 0:
                logger.debug("Rate limiting %s for %.2fs", domain, wait_time)
                await asyncio.sleep(wait_time)
            self._last_request[domain] = time.monotonic()
