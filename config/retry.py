"""Shared retry-with-backoff decorator for transient HTTP failures."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def async_retry_on_transient(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry an async callable on HTTP 429/5xx or transient network errors."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
                        raise
                    last_exc = exc
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "HTTP %s on attempt %s/%s; retrying in %.1fs",
                        status,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                    if attempt >= max_retries:
                        raise
                    last_exc = exc
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Network error on attempt %s/%s: %s; retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Retry loop exited unexpectedly")

        return wrapper

    return decorator
