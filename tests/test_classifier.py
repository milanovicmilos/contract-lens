from typing import Dict

from src.application.interfaces.iclassifier import IClassifier


class MockClassifier(IClassifier):
    """A mock classifier for testing the application logic without ML models."""

    def classify(self, text: str) -> Dict[str, float]:
        if "terminate" in text.lower():
            return {"Termination": 0.95, "Governing Law": 0.05, "Limitation of Liability": 0.01}
        return {"Termination": 0.01, "Governing Law": 0.90, "Limitation of Liability": 0.05}


def test_mock_classifier_interface():
    """Test that the IClassifier contract is maintained and works as expected."""
    classifier = MockClassifier()

    # Test termination clause
    result = classifier.classify("This agreement shall terminate upon 30 days notice.")
    assert result["Termination"] > 0.9
    assert result["Governing Law"] < 0.1

    # Test governing law clause
    result2 = classifier.classify("The laws of the State of New York shall govern.")
    assert result2["Governing Law"] > 0.8
    assert result2["Termination"] < 0.1
