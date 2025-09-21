"""Application logging configuration utilities."""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig


def _resolve_log_level() -> str:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level not in logging._nameToLevel:  # type: ignore[attr-defined]
        return "INFO"
    return level


def setup_logging() -> None:
    """Initialise structured logging for the FastAPI app."""

    log_level = _resolve_log_level()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": log_level},
                "uvicorn.error": {"handlers": ["default"], "level": log_level, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": log_level, "propagate": False},
            },
        }
    )

    logging.getLogger(__name__).debug("Logging initialised at %s level", log_level)

