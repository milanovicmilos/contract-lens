"""
Prometheus metrics for ContractLens.

Counters / histograms live as module-level globals so prometheus-client
registers them exactly once per process. The /metrics endpoint is wired
in src/api/main.py and bypasses both auth and rate-limit middleware so
Prometheus scrapers can pull without a key.
"""

from __future__ import annotations

import os
import time
from typing import Awaitable, Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def metrics_enabled() -> bool:
    raw = (os.getenv("METRICS_ENABLED", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# HTTP-level metrics. Job-, orchestrator-, and LLM-level counters will be
# added in follow-up PRs that own the wiring code (PR-G LLM rewriter,
# PR-H integration tests against real models). We don't declare empty
# counters here because that ships unused symbols.
http_requests_total = Counter(
    "contractlens_http_requests_total",
    "Count of HTTP requests handled.",
    ("method", "path", "status"),
)
http_request_duration_seconds = Histogram(
    "contractlens_http_request_duration_seconds",
    "Wall-clock latency per HTTP request, server-side.",
    ("method", "path"),
    # Buckets are tuned for sub-second API responses; ML inference paths
    # sometimes exceed 1 s so we keep a long tail.
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record HTTP method / path / status / latency for every request.

    Path is taken from the routing template (e.g. ``/api/v1/jobs/{job_id}``)
    so high-cardinality job ids do not blow up Prometheus label space.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not metrics_enabled():
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Prefer the route template when available; fall back to the raw path
        # for unmatched / 404 requests (a small label-cardinality risk we
        # accept because 404 traffic should be low in a healthy service).
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)

        http_requests_total.labels(
            method=request.method,
            path=path_label,
            status=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            path=path_label,
        ).observe(duration)
        return response


def render_metrics() -> Response:
    """Return the Prometheus text exposition for the current registry."""
    body = generate_latest()
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)
