"""Tests for entailment checker."""

from __future__ import annotations

from validation.entailment_checker import EntailmentChecker


class FakePipeline:
    def __call__(self, text):
        return [{"label": "ENTAILMENT", "score": 0.95}]


def test_entailment_checker(monkeypatch) -> None:
    checker = EntailmentChecker()
    monkeypatch.setattr(checker, "_pipeline", FakePipeline())
    label = checker.check(
        "Python was created by Guido van Rossum.",
        "Python is a language created by Guido van Rossum.",
    )
    assert label == "entailment"
