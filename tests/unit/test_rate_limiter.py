"""Tests for domain rate limiter."""

from __future__ import annotations

import time

import pytest

from crawler.rate_limiter import DomainRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_delays() -> None:
    limiter = DomainRateLimiter(requests_per_second=5.0)
    start = time.monotonic()
    await limiter.acquire("example.com")
    await limiter.acquire("example.com")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1
