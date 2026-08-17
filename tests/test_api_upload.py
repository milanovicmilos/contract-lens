"""
Tests for the async upload flow: POST /api/v1/contracts + GET /api/v1/jobs/{id}.

We run the FastAPI app through TestClient (which drives lifespan), set
JOBS_DB_PATH to a tmp file, and override get_orchestrator with a mock
that returns a single deterministic RiskScore. The ThreadPoolExecutor
inside the app is real — it actually runs the upload_worker — so the
test exercises the whole job-tracking path end to end.
"""

from __future__ import annotations

import importlib
import time
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
            justification="Liability cap of $10,000 detected.",
            extracted_span="Liability is capped at $10,000.",
            metadata={"classifier_confidence": 0.81},
        )
    ]
    return mock


@pytest.fixture
def client(tmp_path, monkeypatch, mock_orchestrator):
    """TestClient with a fresh on-disk SQLite job store and a mocked orchestrator."""
    monkeypatch.setenv("TESTING", "True")
    monkeypatch.setenv("API_AUTH_DISABLED", "1")
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("RATE_LIMIT_UPLOAD", "1000/minute")  # don't trip in tests
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "5")

    import src.api.main as api_main
    import src.api.middleware as api_mw
    import src.api.rate_limit as api_rl
    import src.api.security as api_sec

    for mod in (api_sec, api_mw, api_rl, api_main):
        importlib.reload(mod)

    api_main.app.dependency_overrides[api_main.get_orchestrator] = lambda: mock_orchestrator
    with TestClient(api_main.app) as tc:
        yield tc
    api_main.app.dependency_overrides.clear()


def _wait_for(client: TestClient, job_id: str, terminal: tuple[str, ...] = ("completed", "failed")):
    """Poll the job until it reaches a terminal status (or 5s timeout)."""
    deadline = time.time() + 5.0
    while time.time() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in terminal:
            return body
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} did not terminate within 5s; last status = {body['status']}"
    )


def test_upload_returns_202_with_job_id(client):
    """A clean TXT upload returns 202 + a polling URL."""
    files = {
        "file": (
            "demo.txt",
            b"This is a real contract clause that is more than eight words long.",
            "text/plain",
        )
    }
    r = client.post("/api/v1/contracts", files=files)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert body["status_url"] == f"/api/v1/jobs/{body['job_id']}"


def test_upload_completes_and_returns_risk_scores(client):
    """The background worker should produce the mocked RiskScore and the GET surfaces it."""
    # Need > 20 words so the worker does not skip the chunk
    # (matches the analyzer's min_words_per_chunk).
    body_text = (
        "The Service Provider may terminate this agreement for convenience with "
        "thirty days written notice to the other party. Liability of the Service "
        "Provider is capped at ten thousand United States dollars under this contract."
    )
    files = {"file": ("demo.txt", body_text.encode(), "text/plain")}
    accepted = client.post("/api/v1/contracts", files=files).json()

    final = _wait_for(client, accepted["job_id"])
    assert final["status"] == "completed"
    assert final["error"] is None
    assert final["result"], "expected non-empty result list"
    assert final["result"][0]["category"] == "Cap On Liability"
    assert final["source_filename"] == "demo.txt"
    assert final["source_format"] == "txt"


def test_upload_rejects_unsupported_extension(client):
    files = {"file": ("malicious.exe", b"MZ\x90\x00", "application/octet-stream")}
    r = client.post("/api/v1/contracts", files=files)
    assert r.status_code == 415
    assert "unsupported" in r.json()["detail"].lower()


def test_upload_rejects_empty_file(client):
    files = {"file": ("blank.txt", b"", "text/plain")}
    r = client.post("/api/v1/contracts", files=files)
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_jobs_endpoint_returns_404_for_unknown_id(client):
    r = client.get("/api/v1/jobs/does-not-exist")
    assert r.status_code == 404


def test_failed_job_records_error(client, mock_orchestrator):
    """If the orchestrator raises, the job row must persist the exception text."""
    mock_orchestrator.analyze.side_effect = RuntimeError("classifier blew up")
    # Body must clear min_words_per_chunk (20) so the orchestrator is actually called.
    files = {
        "file": (
            "demo.txt",
            (
                b"This contract clause is intentionally written with enough words to clear "
                b"the validator threshold so the background worker actually calls the "
                b"orchestrator method that we have configured to raise."
            ),
            "text/plain",
        )
    }
    accepted = client.post("/api/v1/contracts", files=files).json()
    final = _wait_for(client, accepted["job_id"])
    assert final["status"] == "failed"
    assert final["result"] is None
    assert "classifier blew up" in (final["error"] or "")
