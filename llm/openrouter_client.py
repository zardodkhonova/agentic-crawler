"""OpenRouter API client implementing the LLMClient protocol."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.retry import async_retry_on_transient
from llm.client import LLMClientError

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Async OpenRouter chat-completions client with retry on transient errors."""

    def __init__(
        self,
        api_key: str,
        model: str,
        http_client: httpx.AsyncClient,
        max_retries: int = 6,
        retry_base_delay: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._http_client = http_client
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request to OpenRouter."""
        if not self._api_key:
            raise LLMClientError("OPENROUTER_API_KEY is not configured")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/agentic-crawler",
            "X-Title": "Agentic Crawler",
        }

        @async_retry_on_transient(self._max_retries, self._retry_base_delay)
        async def _post() -> httpx.Response:
            response = await self._http_client.post(
                OPENROUTER_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response

        try:
            response = await _post()
        except httpx.HTTPError as exc:
            logger.error("OpenRouter request failed: %s", exc)
            raise LLMClientError(f"OpenRouter request failed: {exc}") from exc

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.error("Invalid OpenRouter response: %s", response.text[:500])
            raise LLMClientError("Invalid response from OpenRouter") from exc