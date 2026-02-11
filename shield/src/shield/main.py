"""Uvicorn entry point for SHIELD API server."""

from __future__ import annotations

import logging
import sys

import uvicorn

from shield.config import settings


def _configure_logging() -> None:
    """Configure structured logging based on settings."""
    level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if settings.LOG_FORMAT == "json":
        fmt = (
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"module":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    root.handlers = [handler]


def main() -> None:
    """Start the SHIELD API server."""
    _configure_logging()
    logger = logging.getLogger("shield.main")
    logger.info("Starting SHIELD in %s mode on %s:%d", settings.SHIELD_MODE, settings.API_HOST, settings.API_PORT)

    uvicorn.run(
        "shield.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.SHIELD_MODE == "mock",
    )


if __name__ == "__main__":
    main()
