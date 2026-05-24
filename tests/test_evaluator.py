from unittest.mock import MagicMock

import pytest

from src.application.evaluation.evaluator import LLMEvaluator
from src.domain.risk_score import RiskScore


def test_evaluate_faithfulness_with_mock():
    mock_llm_client = MagicMock()
    mock_llm_client.evaluate.return_value = {
        "faithfulness": 0.85,
        "reasoning": "Reasonable deduction.",
    }

    evaluator = LLMEvaluator(llm_client=mock_llm_client)
    risk_score = RiskScore(
        category="Liability",
        risk_level="Medium",
        score=0.6,
        justification="Standard liability cap.",
        extracted_span="Liability is capped at 1000.",
        metadata={},
    )

    result = evaluator.evaluate_faithfulness(risk_score, "Company liability is capped at 1000 USD.")

    mock_llm_client.evaluate.assert_called_once()
    assert result.metric_name == "faithfulness"
    assert result.score == 0.85
    assert result.reasoning == "Reasonable deduction."


def test_evaluate_faithfulness_missing_client():
    evaluator = LLMEvaluator(llm_client=None)
    risk_score = RiskScore(
        category="Liability",
        risk_level="Medium",
        score=0.6,
        justification="Standard liability cap.",
        extracted_span="Liability is capped at 1000.",
        metadata={},
    )

    with pytest.raises(
        ValueError, match="LLM client requires a valid instance for faithfulness evaluation."
    ):
        evaluator.evaluate_faithfulness(risk_score, "Company liability is capped at 1000 USD.")
