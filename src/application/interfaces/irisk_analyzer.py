"""
Interface for Risk Analysis and classification engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RiskScore:
    """
    Represents the risk level and justification for a specific clause.

    Traceability fields (span_start_offset, span_end_offset, source_doc) enable
    auditing every Risk back to the original contract text — a hard requirement
    from the project's Transparency goal: every Risk Score must be backed by a
    citation and a reference to the applicable text location.
    """

    category: str
    risk_level: str  # "Low", "Medium", "High"
    score: float  # 0.0 to 1.0 confidence or exact numerical score
    justification: str
    extracted_span: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Traceability (optional; default to None for backward compatibility with
    # callers that don't yet plumb offsets through).
    span_start_offset: Optional[int] = None
    span_end_offset: Optional[int] = None
    source_doc: Optional[str] = None


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
