from unittest.mock import MagicMock

from src.application.analyze_contract_use_case import AnalyzeContractUseCase
from src.domain.risk_policy import RiskPolicy


def test_analyze_contract_use_case_flow():
    """Verify that AnalyzeContractUseCase orchestrates classifier, extractor, and policy correctly."""
    # 1. Mocks
    mock_classifier = MagicMock()
    mock_extractor = MagicMock()

    # Configure mock classifier to simulate multi-label outputs
    def mock_classify(text):
        if "terminate" in text.lower():
            return {"Termination": 0.95, "Governing Law": 0.1}
        elif "law" in text.lower():
            return {"Termination": 0.05, "Governing Law": 0.85}
        return {"Termination": 0.1, "Governing Law": 0.05}

    mock_classifier.classify.side_effect = mock_classify

    # 2. Setup domain policy
    policy = RiskPolicy(
        rules={
            "Termination": {
                "high_risk_keywords": ["unilateral", "without cause"],
                "default_risk": "Medium",
            },
            "Governing Law": {"high_risk_keywords": ["russia", "unknown"], "default_risk": "Low"},
        }
    )

    # 3. Instantiate Use Case
    use_case = AnalyzeContractUseCase(
        classifier=mock_classifier,
        extractor=mock_extractor,
        risk_policy=policy,
        classification_threshold=0.5,
    )

    # 4. Execute with blocks
    blocks = [
        "This agreement shall terminate without cause upon 30 days notice.",
        "The governing law is the State of New York.",
        "Just a random operational sentence.",
    ]

    results = use_case.execute(blocks)

    # 5. Assertions
    assert len(results) == 2  # Only two blocks break the 0.5 threshold

    # Check Termination result
    termination_result = next(r for r in results if r.category == "Termination")
    assert (
        termination_result.risk_level == "High"
    )  # Because "without cause" matched high risk keyword
    assert termination_result.metadata["block_id"] == 0
    assert termination_result.metadata["classification_confidence"] == 0.95

    # Check Governing Law result
    law_result = next(r for r in results if r.category == "Governing Law")
    assert law_result.risk_level == "Low"
