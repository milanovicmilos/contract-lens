"""
Rate-limit configuration for ContractLens, backed by slowapi.

Per-endpoint limits live next to the route decorator (e.g.
``@limiter.limit("60/minute")``); this module owns construction of the
limiter singleton + the key function so we can attribute requests to the
authenticated API key when present, falling back to the client IP for
unauthenticated / dev-mode traffic.
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _key(request: Request) -> str:
    """Prefer the API key (set by `require_api_key`) over the source IP."""
    forwarded = request.headers.get("x-api-key")
    if forwarded:
        return f"key:{forwarded}"
    return f"ip:{get_remote_address(request)}"


def _default_limits() -> list:
    """slowapi expects a list — empty list means no global default."""
    raw = os.getenv("RATE_LIMIT_DEFAULTS", "120/minute").strip()
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


limiter = Limiter(
    key_func=_key,
    default_limits=_default_limits(),
    headers_enabled=True,
)
