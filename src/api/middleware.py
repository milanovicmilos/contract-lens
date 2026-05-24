"""
ASGI middleware for ContractLens: request size enforcement + request-id propagation.

Body size limit lives here because FastAPI / Starlette do not enforce one
out of the box, and clients can post arbitrarily large contract bodies.

The X-Request-ID middleware assigns a UUID4 to each request (or honours a
client-supplied one) and echoes it on the response, so logs / RAGAS
reports can be correlated end to end.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_MB = 5
REQUEST_ID_HEADER = "X-Request-ID"


def max_body_bytes() -> int:
    """Read MAX_REQUEST_BODY_MB at call time so tests can override via env."""
    return int(os.getenv("MAX_REQUEST_BODY_MB", str(DEFAULT_MAX_BODY_MB))) * 1024 * 1024


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared Content-Length exceeds MAX_REQUEST_BODY_MB.

    We trust Content-Length because Starlette / uvicorn already enforce it
    against the streamed body; spoofing it shorter cuts the client off, and
    spoofing it longer trips this check.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                return JSONResponse({"detail": "Malformed Content-Length header."}, status_code=400)
            cap = max_body_bytes()
            if size > cap:
                return JSONResponse(
                    {
                        "detail": (
                            f"Request body too large ({size} bytes); "
                            f"limit is {cap} bytes (MAX_REQUEST_BODY_MB)."
                        )
                    },
                    status_code=413,
                )
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign / honour X-Request-ID; expose it on request.state and response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response
