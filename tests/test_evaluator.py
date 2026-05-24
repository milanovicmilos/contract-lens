from unittest.mock import MagicMock

import pytest

from src.application.evaluation.evaluator import LLMEvaluator
from src.application.interfaces.illm_provider import LLMResponse
from src.domain.risk_score import RiskScore


def _risk():
    return RiskScore(
        category="Liability",
        risk_level="Medium",
        score=0.6,
        justification="Standard liability cap.",
        extracted_span="Liability is capped at 1000.",
        metadata={},
    )


def test_evaluate_faithfulness_parses_provider_json():
    """Provider returns a JSON blob; evaluator parses it into an EvaluationResult."""
    provider = MagicMock()
    provider.chat.return_value = LLMResponse(
        content='{"faithfulness": 0.85, "reasoning": "Reasonable deduction."}',
        model="fake",
    )

    evaluator = LLMEvaluator(llm_provider=provider)
    result = evaluator.evaluate_faithfulness(_risk(), "Company liability is capped at 1000 USD.")

    provider.chat.assert_called_once()
    assert result.metric_name == "faithfulness"
    assert result.score == 0.85
    assert result.reasoning == "Reasonable deduction."


def test_evaluator_requires_provider():
    """Constructing without a provider must fail loudly — there is no silent fallback."""
    with pytest.raises(ValueError, match="llm_provider is required"):
        LLMEvaluator(llm_provider=None)
