"""Centralized structured logging for the AgentGuard research platform.

Provides a single, consistent log format across all backend and ML modules.
No sensitive or credential information should ever be passed to any logger.

Usage:
    from backend.app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Database initialized successfully")
"""

import logging
import sys
from typing import Optional


# Log format matches research reproducibility requirements:
# ISO-8601 timestamp | level | module path | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_root_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with the standard AgentGuard format.

    Should be called once at application startup. Subsequent calls are
    idempotent — the root logger will only be configured once.

    Args:
        level: Log level string: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    """
    global _root_configured
    if _root_configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(handler)

    # Suppress overly verbose third-party loggers at startup
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _root_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger using the AgentGuard logging hierarchy.

    Args:
        name: Module name, typically ``__name__``. Defaults to root logger.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name or "agentguard")
