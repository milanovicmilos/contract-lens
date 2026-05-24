import os

os.environ["TESTING"] = "True"

from unittest.mock import MagicMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app, get_orchestrator  # noqa: E402
from src.domain.risk_score import RiskScore  # noqa: E402


def _mock_get_orchestrator():
    mock_orch = MagicMock()
    mock_orch.analyze.return_value = [
        RiskScore(
            category="Cap On Liability",
            risk_level="Medium",
            score=0.6,
            justification="Standard liability logic.",
            extracted_span="Liability is limited.",
            metadata={"classifier_confidence": 0.85},
            span_start_offset=10,
            span_end_offset=31,
            source_doc="test.pdf",
        )
    ]
    return mock_orch


app.dependency_overrides[get_orchestrator] = _mock_get_orchestrator
client = TestClient(app)


def test_health_check_responds():
    """Health endpoint must return 200 with version metadata regardless of mode."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "1.0.0"
    assert "status" in body
    assert "orchestrator_ready" in body


def test_analyze_contract_empty_text():
    response = client.post("/api/v1/analyze", json={"text": "   "})
    assert response.status_code == 400


def test_analyze_contract_valid_structure():
    response = client.post(
        "/api/v1/analyze",
        json={"text": "This is a valid contract clause.", "source_doc": "test.pdf"},
    )
    assert response.status_code == 200
    res_list = response.json()
    assert len(res_list) == 1
    item = res_list[0]
    assert item["category"] == "Cap On Liability"
    assert item["risk_level"] == "Medium"
    assert item["span_start_offset"] == 10
    assert item["span_end_offset"] == 31
    assert item["source_doc"] == "test.pdf"
