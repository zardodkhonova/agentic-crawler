"""Tests for claim extractor."""

from __future__ import annotations

import json

import pytest

from validation.claim_extractor import ClaimExtractor


class FakeLLM:
    async def complete(self, prompt, system=None, json_mode=False) -> str:
        return json.dumps({"claims": ["Python was created by Guido van Rossum."]})


@pytest.mark.asyncio
async def test_claim_extractor() -> None:
    extractor = ClaimExtractor(FakeLLM())
    claims = await extractor.decompose("Python was created by Guido van Rossum.")
    assert len(claims) == 1
    assert "Guido" in claims[0].text
