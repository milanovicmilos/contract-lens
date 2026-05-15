"""
Interface for Extractive QA models in the Application layer.
This ensures the domain/application has no hard dependency on HuggingFace or transformers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExtractionResult:
    """Represents a single extracted span from the model."""

    text: str
    answer_start: int
    answer_end: int
    score: float
    question: str
    metadata: Dict[str, Any]


class IExtractor(ABC):
    """
    Abstract interface for Contract Extractors.
    Any AI model (Local DeBERTa, OpenAI, etc.) must implement this interface
    to be used in the Application use-cases.
    """

    @abstractmethod
    def extract(self, context: str, question: str, top_k: int = 1) -> List[ExtractionResult]:
        """
        Extracts top_k spans from the context that answer the question.

        Args:
            context: The contract text or paragraph chunk.
            question: The clause category question.
            top_k: Number of highest-scoring spans to return.

        Returns:
            List of ExtractionResult objects.
        """
        pass

    @abstractmethod
    def batch_extract(
        self, contexts: List[str], question: str, top_k: int = 1
    ) -> List[List[ExtractionResult]]:
        """
        Extracts spans for multiple contexts simultaneously for better throughput.
        """
        pass
