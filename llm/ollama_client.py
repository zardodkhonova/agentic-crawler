"""Ollama local LLM client implementing the LLMClient protocol."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.retry import async_retry_on_transient
from llm.client import LLMClientError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for a locally running Ollama server."""

    def __init__(
        self,
        model: str,
        http_client: httpx.AsyncClient,
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        self._model = model
        self._http_client = http_client
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request to a local Ollama server."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        @async_retry_on_transient(self._max_retries, self._retry_base_delay)
        async def _post() -> httpx.Response:
            response = await self._http_client.post(
                self._url,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return response

        try:
            response = await _post()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama request failed: %s | body: %s",
                exc,
                exc.response.text[:500],
            )
            raise LLMClientError(f"Ollama request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.error(
                "Ollama request failed (is Ollama running? try 'ollama serve'): %s",
                exc,
            )
            raise LLMClientError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
            return data["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.error("Invalid Ollama response: %s", response.text[:500])
            raise LLMClientError("Invalid response from Ollama") from exc