"""
Tests for the API security layer: auth, rate limit, CORS, body size, request ID.

Each test isolates env state in a fixture so the global FastAPI app picks
up the per-test configuration. We rebuild a TestClient per test rather
than reusing the module-level one because middleware reads env on import.
"""

import importlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.domain.risk_score import RiskScore


@pytest.fixture
def mock_orchestrator():
    mock = MagicMock()
    mock.analyze.return_value = [
        RiskScore(
            category="Cap On Liability",
            risk_level="Medium",
            score=0.6,
            justification="Standard liability cap.",
            extracted_span="Liability is capped at $1000.",
            metadata={"classifier_confidence": 0.85},
        )
    ]
    return mock


@pytest.fixture
def configured_app(monkeypatch, mock_orchestrator):
    """Build a TestClient with auth + rate limit configured via env."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.setenv("CONTRACTLENS_API_KEYS", "test-key-A,test-key-B")
    monkeypatch.delenv("API_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("RATE_LIMIT_ANALYZE", "1000/minute")  # don't trip in normal tests
    monkeypatch.setenv("RATE_LIMIT_REPORT", "1000/minute")
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "1")

    # Force-reload the api modules so module-level env reads pick up our patches.
    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    for mod in (api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    api_main.app.dependency_overrides[api_main.get_orchestrator] = lambda: mock_orchestrator
    client = TestClient(api_main.app)
    yield client
    api_main.app.dependency_overrides.clear()


def test_analyze_rejects_missing_api_key(configured_app):
    """Without X-API-Key the analyze endpoint must respond 401."""
    r = configured_app.post("/api/v1/analyze", json={"text": "anything goes here"})
    assert r.status_code == 401
    assert "Missing" in r.json()["detail"]


def test_analyze_rejects_wrong_api_key(configured_app):
    r = configured_app.post(
        "/api/v1/analyze",
        json={"text": "anything goes here"},
        headers={"X-API-Key": "totally-wrong"},
    )
    assert r.status_code == 401
    assert "Invalid" in r.json()["detail"]


def test_analyze_accepts_either_configured_key(configured_app):
    """Both keys in CONTRACTLENS_API_KEYS must be honoured."""
    for key in ("test-key-A", "test-key-B"):
        r = configured_app.post(
            "/api/v1/analyze",
            json={"text": "a real clause that is at least eight words long"},
            headers={"X-API-Key": key},
        )
        assert r.status_code == 200, f"key {key!r} unexpectedly rejected"


def test_health_endpoint_does_not_require_auth(configured_app):
    """Liveness probes must not need a key — kubelet won't have one."""
    r = configured_app.get("/health")
    assert r.status_code == 200


def test_request_id_header_round_trip(configured_app):
    """Server should echo any client-supplied X-Request-ID."""
    r = configured_app.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers.get("X-Request-ID") == "abc-123"


def test_request_id_is_assigned_when_missing(configured_app):
    """Server must mint an X-Request-ID when the client omits one."""
    r = configured_app.get("/health")
    assert r.headers.get("X-Request-ID")  # non-empty
    assert len(r.headers["X-Request-ID"]) >= 16  # uuid4.hex is 32 chars


def test_body_size_limit_rejects_oversize_request(configured_app):
    """A body declaring more than MAX_REQUEST_BODY_MB must get 413."""
    # MAX_REQUEST_BODY_MB=1 in the fixture; send a 2 MB declared body.
    oversize = "x" * (2 * 1024 * 1024)
    r = configured_app.post(
        "/api/v1/analyze",
        json={"text": oversize},
        headers={"X-API-Key": "test-key-A"},
    )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


def test_auth_disabled_mode_bypasses_key_requirement(monkeypatch, mock_orchestrator):
    """API_AUTH_DISABLED=1 must let unauthenticated requests through (dev / CI)."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.setenv("API_AUTH_DISABLED", "1")
    monkeypatch.delenv("CONTRACTLENS_API_KEYS", raising=False)

    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    for mod in (api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    api_main.app.dependency_overrides[api_main.get_orchestrator] = lambda: mock_orchestrator
    client = TestClient(api_main.app)
    r = client.post(
        "/api/v1/analyze",
        json={"text": "a real clause that is at least eight words long"},
    )
    assert r.status_code == 200, r.text
    api_main.app.dependency_overrides.clear()


def test_misconfigured_server_returns_503_when_no_keys(monkeypatch, mock_orchestrator):
    """If auth is on but CONTRACTLENS_API_KEYS is empty, fail closed with 503."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.delenv("API_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("CONTRACTLENS_API_KEYS", "")

    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    for mod in (api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    api_main.app.dependency_overrides[api_main.get_orchestrator] = lambda: mock_orchestrator
    client = TestClient(api_main.app)
    r = client.post(
        "/api/v1/analyze",
        json={"text": "anything"},
        headers={"X-API-Key": "some-key"},
    )
    assert r.status_code == 503
    api_main.app.dependency_overrides.clear()


def test_rate_limit_returns_429_after_quota_exhausted(monkeypatch, mock_orchestrator):
    """slowapi should reject the (N+1)th request inside the quota window."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.setenv("CONTRACTLENS_API_KEYS", "test-key")
    monkeypatch.delenv("API_AUTH_DISABLED", raising=False)
    # Tight quota so the test runs fast.
    monkeypatch.setenv("RATE_LIMIT_ANALYZE", "2/minute")
    monkeypatch.setenv("RATE_LIMIT_DEFAULTS", "")  # don't add a global cap on top

    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    for mod in (api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    api_main.app.dependency_overrides[api_main.get_orchestrator] = lambda: mock_orchestrator
    client = TestClient(api_main.app)
    headers = {"X-API-Key": "test-key"}
    payload = {"text": "a real clause that is at least eight words long"}

    # First two requests fit the 2/minute quota.
    assert client.post("/api/v1/analyze", json=payload, headers=headers).status_code == 200
    assert client.post("/api/v1/analyze", json=payload, headers=headers).status_code == 200

    # Third request must be 429.
    r3 = client.post("/api/v1/analyze", json=payload, headers=headers)
    assert r3.status_code == 429
    assert "Rate limit" in r3.json()["detail"]

    api_main.app.dependency_overrides.clear()
