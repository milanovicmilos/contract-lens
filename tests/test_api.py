import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.main import app, get_orchestrator
from src.application.interfaces.irisk_analyzer import RiskScore

client = TestClient(app)

def mock_get_orchestrator():
    mock_orch = MagicMock()
    mock_orch.analyze.return_value = [
        RiskScore(
            category="Liability",
            risk_level="Medium",
            score=0.6,
            justification="Standard liability logic.",
            extracted_span="Liability is limited.",
            metadata={}
        )
    ]
    return mock_orch

app.dependency_overrides[get_orchestrator] = mock_get_orchestrator

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

def test_analyze_contract_empty_text():
    response = client.post("/api/v1/analyze", json={"text": "   "})
    assert response.status_code == 400

def test_analyze_contract_valid_structure():
    response = client.post("/api/v1/analyze", json={"text": "This is a valid contract clause."})
    assert response.status_code == 200
    res_list = response.json()
    assert len(res_list) == 1
    assert res_list[0]["category"] == "Liability"
    assert res_list[0]["risk_level"] == "Medium"
