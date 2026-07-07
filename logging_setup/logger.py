"""Centralized logging configuration."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure console and rotating file logging via dictConfig."""
    log_path = settings.log_path
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger(__name__).warning("Could not create log directory: %s", exc)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "standard",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "standard",
                "filename": str(log_path),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(config)
