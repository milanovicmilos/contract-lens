"""
Unit tests for KeywordClassifier — the regex baseline.

Strategy: test that canonical positive clauses trigger the right category,
clear negatives do not, and the IClassifier interface contract is satisfied.
"""

from __future__ import annotations

import pytest

from src.infrastructure.ai.hf_classifier import CUAD_CATEGORIES
from src.infrastructure.ai.keyword_classifier import KeywordClassifier


@pytest.fixture
def clf():
    return KeywordClassifier()


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


def test_returns_dict(clf):
    result = clf.classify("This Agreement shall be governed by the laws of Delaware.")
    assert isinstance(result, dict)


def test_returns_all_41_categories(clf):
    result = clf.classify("Some generic clause text for testing purposes only.")
    assert set(result.keys()) == set(CUAD_CATEGORIES)


def test_empty_text_returns_empty_dict(clf):
    assert clf.classify("") == {}
    assert clf.classify("   ") == {}


def test_all_scores_are_float(clf):
    result = clf.classify("License grant for unlimited perpetual use.")
    for v in result.values():
        assert isinstance(v, float), f"score for key must be float, got {type(v)}"


def test_scores_are_binary_01(clf):
    """KeywordClassifier returns exactly 0.0 or 1.0 — no probabilities."""
    result = clf.classify("The aggregate liability of each party shall not exceed $1,000,000.")
    for v in result.values():
        assert v in (0.0, 1.0), f"Expected 0.0 or 1.0, got {v}"


# ---------------------------------------------------------------------------
# Canonical positive examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,clause",
    [
        (
            "Cap On Liability",
            "The aggregate liability of each party under this Agreement shall not exceed "
            "one million dollars ($1,000,000) per calendar year.",
        ),
        (
            "Governing Law",
            "This Agreement shall be governed by and construed in accordance with the "
            "laws of the State of Delaware.",
        ),
        (
            "Non-Compete",
            "During the Term and for two years thereafter, the Employee shall not directly "
            "or indirectly engage in any business that competes with the Company.",
        ),
        (
            "License Grant",
            "Licensor hereby grants to Licensee a non-exclusive, worldwide license to "
            "use, reproduce, and distribute the Licensed Software.",
        ),
        (
            "Anti-Assignment",
            "Neither party shall assign this Agreement or any of its rights without the "
            "prior written consent of the other party.",
        ),
        (
            "Termination For Convenience",
            "Either party may terminate this Agreement for convenience upon thirty days' "
            "written notice without cause.",
        ),
        (
            "Exclusivity",
            "The Distributor is appointed as the sole and exclusive distributor in the "
            "Territory for the Products.",
        ),
        (
            "Irrevocable Or Perpetual License",
            "Licensor grants Licensee an irrevocable, perpetual license to use the "
            "licensed materials in perpetuity.",
        ),
        (
            "Insurance",
            "Each party shall maintain general commercial liability insurance with "
            "coverage of not less than $2,000,000 per occurrence.",
        ),
        (
            "Audit Rights",
            "Company shall have the right to audit Supplier's books and records upon "
            "reasonable prior notice.",
        ),
        (
            "Rofr/Rofo/Rofn",
            "Company shall have the right of first refusal to purchase any shares "
            "offered by the Stockholder.",
        ),
    ],
)
def test_canonical_positive_clause_triggers_category(clf, category, clause):
    result = clf.classify(clause)
    assert (
        result[category] == 1.0
    ), f"Expected '{category}' = 1.0 for canonical clause: {clause[:80]}…"


# ---------------------------------------------------------------------------
# Clear negative examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,unrelated_clause",
    [
        (
            "Cap On Liability",
            "The parties agree to resolve all disputes through binding arbitration "
            "administered by the American Arbitration Association.",
        ),
        (
            "Non-Compete",
            "Licensee shall pay a quarterly royalty of five percent of net revenues "
            "derived from distribution of the Licensed Product.",
        ),
        (
            "Source Code Escrow",
            "This Agreement shall be governed by the laws of the State of California "
            "without regard to conflicts of law principles.",
        ),
    ],
)
def test_unrelated_clause_does_not_trigger_category(clf, category, unrelated_clause):
    result = clf.classify(unrelated_clause)
    assert (
        result[category] == 0.0
    ), f"'{category}' should be 0.0 for unrelated clause: {unrelated_clause[:80]}…"


# ---------------------------------------------------------------------------
# Custom pattern override
# ---------------------------------------------------------------------------


def test_custom_patterns_override(clf):
    """Constructor accepts a custom pattern list, enabling extension without subclassing."""
    custom_clf = KeywordClassifier(patterns=[("Cap On Liability", [r"\bLIMIT\b"])])
    result = custom_clf.classify("The LIMIT shall apply to all damages.")
    assert result["Cap On Liability"] == 1.0
    # Only 1 category should be in the result (only what was defined)
    assert list(result.keys()) == ["Cap On Liability"]


def test_custom_patterns_miss_when_no_match(clf):
    custom_clf = KeywordClassifier(patterns=[("Cap On Liability", [r"\bXYZZY\b"])])
    result = custom_clf.classify("Liability shall not exceed one million dollars.")
    assert result["Cap On Liability"] == 0.0
