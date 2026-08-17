from abc import ABC, abstractmethod
from typing import Dict


class IClassifier(ABC):
    """
    Interface for the multi-label clause classifier.
    Ensures that the application layer is decoupled from the machine learning infrastructure.
    """

    @abstractmethod
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classifies a given text and returns risk labels with confidence scores.

        Args:
            text: The clause or paragraph text to classify.

        Returns:
            Dictionary mapping label names to probability scores (0.0 to 1.0).
        """
        pass
