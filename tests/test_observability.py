"""
Tests for structured logging + Prometheus /metrics.

Exercises both wires end-to-end through a real FastAPI TestClient so the
middleware order and lifespan setup are part of what gets tested.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """TestClient with a fresh app build, auth disabled, metrics enabled."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.setenv("API_AUTH_DISABLED", "1")
    monkeypatch.setenv("METRICS_ENABLED", "1")
    monkeypatch.setenv("LOG_FORMAT", "json")

    import src.api.logging_config as api_log
    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    # NB: src.api.metrics is intentionally NOT reloaded — prometheus-client
    # registers Counters / Histograms in a process-global registry, and
    # re-importing the module would raise "Duplicated timeseries".
    for mod in (api_log, api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    with TestClient(api_main.app) as tc:
        yield tc


def test_metrics_endpoint_returns_prometheus_exposition(client):
    """/metrics responds 200 with the prometheus text format."""
    # Drive a real request first so the counter has at least one observation.
    client.get("/health")

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "contractlens_http_requests_total" in body
    assert "contractlens_http_request_duration_seconds" in body
    # The /health hit above should have produced at least one entry
    # for the GET /health path label.
    assert 'path="/health"' in body or "path='/health'" in body


def test_metrics_endpoint_can_be_disabled(monkeypatch):
    """METRICS_ENABLED=0 must turn /metrics into a 404."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.setenv("API_AUTH_DISABLED", "1")
    monkeypatch.setenv("METRICS_ENABLED", "0")

    import src.api.logging_config as api_log
    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    # NB: src.api.metrics is intentionally NOT reloaded — prometheus-client
    # registers Counters / Histograms in a process-global registry, and
    # re-importing the module would raise "Duplicated timeseries".
    for mod in (api_log, api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    with TestClient(api_main.app) as tc:
        r = tc.get("/metrics")
    assert r.status_code == 404


def test_request_id_propagates_to_response(client):
    """The middleware should echo X-Request-ID on every response."""
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")  # non-empty


def test_request_id_is_honoured_when_provided(client):
    r = client.get("/health", headers={"X-Request-ID": "trace-abc"})
    assert r.headers["X-Request-ID"] == "trace-abc"


def test_api_key_fingerprint_not_raw_in_logs():
    """Sanity check: the fingerprint helper must hash, never echo the raw key."""
    from src.api.logging_config import _api_key_fingerprint

    fp = _api_key_fingerprint("super-secret-key-do-not-log")
    assert fp != "super-secret-key-do-not-log"
    assert len(fp) == 10
    assert _api_key_fingerprint(None) is None
    assert _api_key_fingerprint("") is None
