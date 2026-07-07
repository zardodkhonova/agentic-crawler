"""LLM client protocol and shared exceptions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMClientError(Exception):
    """Raised when an LLM provider fails after retries."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for swappable LLM providers."""

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Complete a prompt and return the model response text."""
        ...
