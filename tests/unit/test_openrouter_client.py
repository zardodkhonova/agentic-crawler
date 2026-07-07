"""Tests for OpenRouter LLM client."""

from __future__ import annotations

import json

import httpx
import pytest

from llm.client import LLMClientError
from llm.openrouter_client import OpenRouterClient


@pytest.mark.asyncio
async def test_openrouter_complete_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"choices": [{"message": {"content": "Paris"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = OpenRouterClient("test-key", "test-model", client)
    result = await llm.complete("Capital of France?")
    assert result == "Paris"
    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_missing_api_key() -> None:
    client = httpx.AsyncClient()
    llm = OpenRouterClient("", "test-model", client)
    with pytest.raises(LLMClientError):
        await llm.complete("hello")
    await client.aclose()
