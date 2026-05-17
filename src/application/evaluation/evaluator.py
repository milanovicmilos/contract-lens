from dataclasses import dataclass
from typing import Any

from src.application.interfaces.irisk_analyzer import RiskScore


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    reasoning: str


class LLMEvaluator:
    """Evaluates RAG and Consultant extractions using LLM-as-a-judge patterns."""

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def evaluate_faithfulness(self, risk_score: RiskScore, original_text: str) -> EvaluationResult:
        """
        Evaluates whether the justification (especially Consultant Note) is
        faithful to the original text.
        """
        if not self.llm_client:
            # Fallback for testing without LLM
            return EvaluationResult(
                metric_name="faithfulness", score=1.0, reasoning="Fallback: perfectly faithful."
            )

        evaluation = self.llm_client.evaluate(
            original_text=original_text, justification=risk_score.justification
        )

        return EvaluationResult(
            metric_name="faithfulness",
            score=evaluation.get("faithfulness", 0.0),
            reasoning=evaluation.get("reasoning", "No reasoning provided."),
        )
