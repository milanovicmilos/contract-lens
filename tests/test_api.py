import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

def test_analyze_contract_empty_text():
    response = client.post("/api/v1/analyze", json={"text": "   "})
    assert response.status_code == 400

def test_analyze_contract_valid_structure():
    # Currently returns empty array from the scaffolding
    response = client.post("/api/v1/analyze", json={"text": "This is a valid contract clause."})
    assert response.status_code == 200
    assert response.json() == []
