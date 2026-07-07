"""Tests for logging setup."""

from __future__ import annotations

import logging

from config.settings import Settings
from logging_setup.logger import configure_logging


def test_configure_logging(tmp_path) -> None:
    settings = Settings(log_file=str(tmp_path / "test.log"), log_level="DEBUG")
    configure_logging(settings)
    logger = logging.getLogger("test.logger")
    logger.info("hello")
    assert tmp_path.joinpath("test.log").exists()
