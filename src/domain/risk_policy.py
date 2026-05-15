"""
Policy definitions defining what constitutes a Risk.
"""

from typing import Any, Dict


class RiskPolicy:
    """
    Defines the rules for scoring risks.
    """

    def __init__(self, rules: Dict[str, Any] = None):
        """
        Initializes the Risk Policy.

        Args:
            rules: A dictionary mapping categories to their risk criteria.
        """
        # A basic default policy for demonstration purposes.
        self.rules = rules or {
            "Governing Law": {
                "high_risk_keywords": ["Russia", "China", "Unknown"],
                "default_risk": "Medium",
            },
            "Non-Compete": {
                "high_risk_keywords": ["worldwide", "perpetual", "unlimited"],
                "default_risk": "High",
            },
        }

    def assess_risk(self, category: str, text: str) -> tuple[str, float, str]:
        """
        Calculates the risk level based on simple keyword matching as a baseline.

        Args:
            category: The clause category.
            text: The text to assess.

        Returns:
            Tuple containing (Risk Level, Score, Justification).
        """
        if category not in self.rules:
            return "Low", 0.1, "No specific risk policy defined; assumed low risk."

        policy = self.rules[category]
        text_lower = text.lower()

        for kw in policy.get("high_risk_keywords", []):
            if kw.lower() in text_lower:
                return "High", 0.9, f"High risk keyword '{kw}' detected."

        return policy.get("default_risk", "Medium"), 0.5, "Standard clause terms apply."
