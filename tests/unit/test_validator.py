"""Tests for answer validator."""

from __future__ import annotations

import json

import pytest

from validation.entailment_checker import EntailmentChecker
from validation.claim_extractor import ClaimExtractor
from validation.schemas import SourceChunk
from validation.validator import Validator


class FakeLLM:
    async def complete(self, prompt, system=None, json_mode=False) -> str:
        return json.dumps({"claims": ["Paris is the capital of France."]})


class FakeEntailment:
    def check(self, claim_text, chunk_text) -> str:
        return "entailment"


@pytest.mark.asyncio
async def test_validator_supported_claim() -> None:
    validator = Validator(ClaimExtractor(FakeLLM()), FakeEntailment(), confidence_threshold=0.75)
    result = await validator.validate(
        "Paris is the capital of France.",
        [SourceChunk("c1", "Paris is the capital of France.", "https://example.com")],
    )
    assert result.confidence >= 0.75
    assert not result.unsupported_claims
