"""
Port for the risk-analysis use case.

The concrete RiskScore value object lives in the domain layer
(`src.domain.risk_score`). This module re-exports it for backward
compatibility; new code should import RiskScore directly from
`src.domain.risk_score`.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from src.domain.risk_score import RiskScore

__all__ = ["IRiskAnalyzer", "RiskScore"]


class IRiskAnalyzer(ABC):
    """Abstract port for analyzing risk in contract clauses.

    Separates the domain logic of defining risk from the underlying DL
    classification model that produces candidate clauses.
    """

    @abstractmethod
    def analyze_clause(self, category: str, extracted_text: str) -> RiskScore:
        """
        Analyze an extracted clause to determine its risk level.

        Args:
            category: The contract clause category (e.g., "Governing Law").
            extracted_text: The span of text extracted from the document.

        Returns:
            A RiskScore detailing the risk level and justification.
        """

    @abstractmethod
    def evaluate_contract_risks(self, extracts: Dict[str, List[str]]) -> List[RiskScore]:
        """
        Evaluate a set of extracted clauses for a given contract.

        Args:
            extracts: Dictionary of categorized extracted text spans.
                      Format: {"category_name": ["span1", "span2"]}

        Returns:
            List of RiskScores.
        """
