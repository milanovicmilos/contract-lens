"""
Structured logging setup for ContractLens.

Wires structlog so every log record from the API process carries the
same keys (request_id, api_key_fp, level, logger, event) and renders
either as JSON (default in production) or coloured key=value text
(default in dev / TESTING). Stdlib ``logging`` calls are funneled into
the same processor chain so third-party libraries — uvicorn, slowapi,
chromadb — stay legible without separate config.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Any, Dict, Optional

import structlog


def _api_key_fingerprint(api_key: Optional[str]) -> Optional[str]:
    """Short, non-reversible fingerprint of an API key for log attribution.

    Logging the raw key would leak credentials into log aggregators.
    SHA-256 truncated to 10 hex chars is enough to correlate requests
    without inverting back to the secret.
    """
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:10]


def _format_from_env() -> str:
    raw = (os.getenv("LOG_FORMAT", "") or "").strip().lower()
    if raw in {"json", "text"}:
        return raw
    # default: JSON in containers (no TTY), human-friendly text on dev machines
    return "text" if sys.stderr.isatty() else "json"


def _level_from_env() -> int:
    raw = (os.getenv("LOG_LEVEL", "INFO") or "").strip().upper()
    return getattr(logging, raw, logging.INFO)


_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent — call once at process start (lifespan does this)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = _level_from_env()
    log_format = _format_from_env()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=shared_processors + [structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Funnel stdlib logging (uvicorn, slowapi, chromadb, ...) into the same
    # renderer so log lines have a uniform shape regardless of origin.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    # Make uvicorn / fastapi a bit less chatty by default — they otherwise
    # log every successful request at INFO, which doubles up with our
    # access middleware.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def bind_request_context(*, request_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Attach request_id + hashed api_key to the contextvar so every log emitted
    on this thread/task carries them. Returns the bindings for callers that
    also want them inline.
    """
    bindings: Dict[str, Any] = {"request_id": request_id}
    fp = _api_key_fingerprint(api_key)
    if fp:
        bindings["api_key_fp"] = fp
    structlog.contextvars.bind_contextvars(**bindings)
    return bindings


def clear_request_context() -> None:
    """Remove the request-scoped bindings — call in middleware after the response."""
    structlog.contextvars.clear_contextvars()
