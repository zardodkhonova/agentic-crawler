"""Tests for config settings and retry utilities."""

from __future__ import annotations

import pytest

from config.retry import async_retry_on_transient
from config.settings import Settings


def test_settings_loads_defaults() -> None:
    settings = Settings()
    assert settings.max_pages_per_query == 35
    assert settings.confidence_threshold == 0.70
    assert "llama" in settings.openrouter_model.lower() or settings.openrouter_model


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_try() -> None:
    calls = {"count": 0}

    @async_retry_on_transient(max_retries=2, base_delay=0.01)
    async def ok() -> str:
        calls["count"] += 1
        return "done"

    assert await ok() == "done"
    assert calls["count"] == 1