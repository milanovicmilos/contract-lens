"""
Interface for Risk Analysis and classification engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RiskScore:
    """Represents the risk level and justification for a specific clause."""

    category: str
    risk_level: str  # "Low", "Medium", "High"
    score: float  # 0.0 to 1.0 confidence or exact numerical score
    justification: str
    extracted_span: str
    metadata: Dict[str, Any]


class IRiskAnalyzer(ABC):
    """
    Abstract interface for analyzing risk in contract clauses.
    This separates the domain logic of defining risk from the underlying DL classification model.
    """

    @abstractmethod
    def analyze_clause(self, category: str, extracted_text: str) -> RiskScore:
        """
        Analyzes an extracted clause to determine its risk level.

        Args:
            category: The contract clause category (e.g., "Governing Law").
            extracted_text: The span of text extracted from the document.

        Returns:
            A RiskScore detailing the risk level and justification.
        """
        pass

    @abstractmethod
    def evaluate_contract_risks(self, extracts: Dict[str, List[str]]) -> List[RiskScore]:
        """
        Evaluates a set of extracted clauses for a given contract.

        Args:
            extracts: Dictionary of categorized extracted text spans.
                      Format: {"category_name": ["span1", "span2"]}

        Returns:
            List of RiskScores.
        """
        pass
