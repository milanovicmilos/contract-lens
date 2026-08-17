"""
Unit tests for the Risk Policy domain logic.
"""

from src.domain.risk_policy import RiskPolicy


class TestRiskPolicy:

    def test_assess_risk_high_risk_keyword(self):
        policy = RiskPolicy()
        level, score, just = policy.assess_risk(
            "Governing Law", "This agreement is governed by the laws of Russia."
        )
        assert level == "High"
        assert score == 0.9
        assert "Russia" in just

    def test_assess_risk_default(self):
        policy = RiskPolicy()
        level, score, just = policy.assess_risk(
            "Governing Law", "Governed by the laws of New York."
        )
        assert level == "Medium"
        assert score == 0.5

    def test_assess_risk_unknown_category(self):
        policy = RiskPolicy()
        level, score, just = policy.assess_risk("Some Other Clause", "Blah blah.")
        assert level == "Low"
        assert score == 0.1
