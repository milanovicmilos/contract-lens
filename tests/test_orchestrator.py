from unittest.mock import MagicMock

import pytest

from src.application.interfaces.iextractor import ExtractionResult
from src.application.orchestration.orchestrator import ContractOrchestrator
from src.domain.risk_policy import RiskPolicy


@pytest.fixture
def mock_classifier():
    """Classifier returns a high Termination score when 'terminate' appears."""
    mock = MagicMock()
    mock.classify.side_effect = lambda t: (
        {"Termination For Convenience": 0.9} if "terminate" in t.lower() else {"Other": 0.4}
    )
    return mock


@pytest.fixture
def mock_extractor():
    """Extractor returns a single span anchored at offset 5 for any context/question pair."""
    mock = MagicMock()
    mock.extract.return_value = [
        ExtractionResult(
            text="terminate immediately",
            answer_start=5,
            answer_end=26,
            score=0.85,
            question="placeholder",
            metadata={},
        )
    ]
    return mock


@pytest.fixture
def risk_policy():
    return RiskPolicy(
        rules={
            "Termination For Convenience": {
                "high_risk_keywords": ["immediately"],
                "default_risk": "Medium",
                "rationale": "Test rationale for termination.",
            }
        }
    )


def test_orchestrator_valid_flow_with_extractor(mock_classifier, mock_extractor, risk_policy):
    """End-to-end happy path: extractor is called, span offsets land in RiskScore."""
    orchestrator = ContractOrchestrator(
        classifier=mock_classifier,
        risk_policy=risk_policy,
        extractor=mock_extractor,
        llm_client=None,  # graceful degradation expected
    )

    text = "This contract may terminate immediately upon material breach by either party."
    risks = orchestrator.analyze(text, source_doc="test_contract.pdf")

    assert len(risks) == 1
    risk = risks[0]
    assert risk.category == "Termination For Convenience"
    assert risk.risk_level == "High"  # 'immediately' keyword triggers escalation
    assert risk.extracted_span == "terminate immediately"
    assert risk.span_start_offset == 5
    assert risk.span_end_offset == 26
    assert risk.source_doc == "test_contract.pdf"
    assert "Local rule-based" in risk.justification  # graceful LLM fallback message
    mock_extractor.extract.assert_called_once()


def test_orchestrator_skips_when_header(mock_classifier, mock_extractor, risk_policy):
    """Validator should short-circuit on text that looks like a section header."""
    orchestrator = ContractOrchestrator(
        classifier=mock_classifier,
        risk_policy=risk_policy,
        extractor=mock_extractor,
        llm_client=None,
    )

    risks = orchestrator.analyze("ARTICLE 5. TERMINATION RIGHTS AND OBLIGATIONS")

    assert risks == []


def test_orchestrator_skips_when_too_short(mock_classifier, mock_extractor, risk_policy):
    """Validator should short-circuit when the text has fewer than the minimum word count."""
    orchestrator = ContractOrchestrator(
        classifier=mock_classifier,
        risk_policy=risk_policy,
        extractor=mock_extractor,
        llm_client=None,
    )

    risks = orchestrator.analyze("Short text terminate")

    assert risks == []


def test_orchestrator_skips_extractor_when_not_provided(mock_classifier, risk_policy):
    """When no extractor is supplied, the pipeline still produces a RiskScore using the full text."""
    orchestrator = ContractOrchestrator(
        classifier=mock_classifier,
        risk_policy=risk_policy,
        extractor=None,
        llm_client=None,
    )

    text = "This contract may terminate immediately upon material breach by either party."
    risks = orchestrator.analyze(text)

    assert len(risks) == 1
    assert risks[0].span_start_offset is None
    assert risks[0].extracted_span == text


def test_orchestrator_invokes_llm_when_provided(mock_classifier, mock_extractor, risk_policy):
    """When an LLM client is provided, the consultant analysis should reach the RiskScore."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This clause permits convenience termination with no notice."
    mock_llm.invoke.return_value = mock_response

    orchestrator = ContractOrchestrator(
        classifier=mock_classifier,
        risk_policy=risk_policy,
        extractor=mock_extractor,
        llm_client=mock_llm,
    )

    text = "This contract may terminate immediately upon material breach by either party."
    risks = orchestrator.analyze(text)

    assert len(risks) == 1
    assert "permits convenience termination" in risks[0].justification
    mock_llm.invoke.assert_called_once()
