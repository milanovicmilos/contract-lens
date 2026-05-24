"""
API key authentication for ContractLens.

Single mechanism: an ``X-API-Key`` HTTP header whose value must match one
of the keys configured via the ``CONTRACTLENS_API_KEYS`` env var
(comma-separated). The dependency is mounted on every protected route.

Secure-by-default: if no keys are configured the dependency rejects every
request. The ``API_AUTH_DISABLED`` env var is the explicit opt-out for
local development and CI test runs; when set it logs a loud WARNING at
startup and treats every request as authenticated.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional, Set

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER_NAME = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_keys() -> Set[str]:
    raw = os.getenv("CONTRACTLENS_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def auth_disabled() -> bool:
    """True when API_AUTH_DISABLED is explicitly set (dev / CI only)."""
    return _truthy(os.getenv("API_AUTH_DISABLED"))


def warn_if_insecure() -> None:
    """Emit a loud warning at startup if auth is off or no keys are configured."""
    if auth_disabled():
        logger.warning(
            "API_AUTH_DISABLED=1 — API key authentication is OFF. "
            "Never use this configuration in production."
        )
        return
    if not _load_keys():
        logger.warning(
            "CONTRACTLENS_API_KEYS is empty. The API will reject every authenticated "
            "request. Set CONTRACTLENS_API_KEYS=<comma,separated,keys> or "
            "API_AUTH_DISABLED=1 (local/dev only)."
        )


async def require_api_key(api_key: Optional[str] = Depends(_api_key_scheme)) -> str:
    """FastAPI dependency that validates the X-API-Key header.

    Returns the matched key on success so downstream code (rate limiter,
    audit log) can attribute the request to a specific principal.
    """
    if auth_disabled():
        return "auth-disabled"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER_NAME} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
        )

    valid_keys = _load_keys()
    if not valid_keys:
        # Misconfigured server (no keys loaded). Fail closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured on the server.",
        )

    # Constant-time comparison against every configured key.
    for candidate in valid_keys:
        if secrets.compare_digest(api_key, candidate):
            return candidate

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
        headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
    )
