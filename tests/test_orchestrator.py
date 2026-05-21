from unittest.mock import MagicMock

import pytest

from src.domain.risk_policy import RiskPolicy
from src.infrastructure.agents.orchestrator import ContractOrchestrator


@pytest.fixture
def orchestrator():
    # Setup mocks
    mock_classifier = MagicMock()
    # Let's say if text has 'terminate', we give it a High score for Termination
    mock_classifier.classify.side_effect = lambda t: (
        {"Termination": 0.9} if "terminate" in t.lower() else {"Other": 0.9}
    )

    mock_extractor = MagicMock()
    mock_policy = RiskPolicy(
        rules={"Termination": {"high_risk_keywords": ["immediately"], "default_risk": "Medium"}}
    )

    return ContractOrchestrator(
        classifier=mock_classifier,
        extractor=mock_extractor,
        risk_policy=mock_policy,
        llm_client=None,
    )


def test_orchestrator_valid_flow(orchestrator):
    """Test full LangGraph flow for a valid clause."""
    text = "This contract may terminate immediately. Blah blah blah."  # added words to pass validation (>5)

    risks = orchestrator.analyze(text)

    # State should have navigated correctly to Auditor and populated final risks.
    assert len(risks) == 1
    assert risks[0].category == "Termination"
    assert risks[0].risk_level == "High"  # Triggered 'immediately' keyword
    assert "Consultant Note" in risks[0].justification


def test_orchestrator_invalid_flow_routing(orchestrator):
    """Test standard LangGraph routing for an invalid clause (e.g. too short)."""
    # 3 words, will fail the length > 5 validator heurustic
    text = "Short noisy header"

    risks = orchestrator.analyze(text)

    # Due to Conditional Edge ending early, final_risks is never populated.
    assert len(risks) == 0
