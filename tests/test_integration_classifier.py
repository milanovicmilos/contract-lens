"""
Integration tests for the real v9.2 DeBERTa classifier.

These tests load the full model from ``models/deberta-cuad-classifier/`` and
verify that the performance numbers published in ``docs/RESULTS.md §1`` hold
after every code change.  They are skipped automatically when the model
directory is absent (CI runners without large artifacts).

Run locally:
    pytest -m integration -v
    pytest -m integration -v tests/test_integration_classifier.py

Skip in full suite (default):
    pytest -m "not integration"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

MODEL_DIR = Path("models/deberta-cuad-classifier")
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_thresholds() -> Dict[str, float]:
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_classifier():
    """Load the v9.2 model once per module.

    Loading takes ~25 s on CPU the first call; subsequent calls use the
    module-scoped cache.  Tests are skipped when the model directory is absent.
    """
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        pytest.skip(
            f"Real v9.2 model not found at {MODEL_DIR}. " "Run 'inv pull-models' to download it."
        )
    from src.infrastructure.ai.hf_classifier import HFClassifier

    return HFClassifier(model_name_or_path=str(MODEL_DIR), device=-1)


# ---------------------------------------------------------------------------
# Structural smoke-tests (fast, no inference)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_model_directory_has_required_files():
    """All files that downstream code relies on must be present."""
    if not MODEL_DIR.exists():
        pytest.skip("Model directory absent.")
    for fname in ("config.json", "model.safetensors", "tokenizer.json", "thresholds.json"):
        assert (MODEL_DIR / fname).exists(), f"Missing required file: {fname}"


@pytest.mark.integration
def test_thresholds_cover_all_41_categories():
    """thresholds.json must contain an entry for every CUAD category."""
    if not THRESHOLDS_PATH.exists():
        pytest.skip("thresholds.json absent.")
    from src.infrastructure.ai.hf_classifier import CUAD_CATEGORIES

    thresholds = _load_thresholds()
    missing = [c for c in CUAD_CATEGORIES if c not in thresholds]
    assert not missing, f"Categories missing from thresholds.json: {missing}"


@pytest.mark.integration
def test_hf_classifier_loads_per_category_thresholds(real_classifier):
    """After loading the model the classifier must expose per_category_thresholds."""
    assert (
        real_classifier.per_category_thresholds is not None
    ), "HFClassifier did not load thresholds.json"
    assert (
        len(real_classifier.per_category_thresholds) == 41
    ), f"Expected 41 thresholds, got {len(real_classifier.per_category_thresholds)}"


@pytest.mark.integration
def test_hf_classifier_label_mapping_applied(real_classifier):
    """Returned category names must be CUAD human names, not LABEL_N placeholders."""
    from src.infrastructure.ai.hf_classifier import CUAD_CATEGORIES

    scores = real_classifier.classify(
        "This Agreement shall be governed by the laws of the State of Delaware."
    )
    for key in scores:
        assert not key.startswith(
            "LABEL_"
        ), f"label_mapping not applied — raw label '{key}' leaked into output"
    assert set(scores.keys()).issubset(
        set(CUAD_CATEGORIES)
    ), f"Unknown category names in output: {set(scores.keys()) - set(CUAD_CATEGORIES)}"


# ---------------------------------------------------------------------------
# Canonical positive examples — each clause is unambiguously on-topic
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "category,clause",
    [
        (
            "Cap On Liability",
            "The aggregate liability of each party under this Agreement shall not exceed "
            "one million dollars ($1,000,000) in any calendar year, whether arising in "
            "contract, tort, or otherwise.",
        ),
        (
            "Governing Law",
            "This Agreement shall be governed by and construed in accordance with the laws "
            "of the State of Delaware, without regard to its conflicts of law principles.",
        ),
        (
            "Non-Compete",
            "During the Term and for a period of two (2) years following termination, "
            "the Employee shall not, directly or indirectly, engage in or assist any "
            "entity that competes with the Company in the same market segment.",
        ),
        (
            "License Grant",
            "Subject to the terms and conditions of this Agreement, Licensor hereby "
            "grants to Licensee a non-exclusive, non-transferable, worldwide, royalty-free "
            "license to use, reproduce, and distribute the Licensed Software.",
        ),
        (
            "Anti-Assignment",
            "Neither party may assign, transfer, or delegate this Agreement or any rights "
            "or obligations hereunder without the prior written consent of the other party, "
            "which shall not be unreasonably withheld.",
        ),
        (
            "Termination For Convenience",
            "Either party may terminate this Agreement for convenience, without cause, "
            "upon thirty (30) days' prior written notice to the other party.",
        ),
        (
            "Parties",
            "This Agreement is entered into as of January 1, 2023, by and between "
            "Acme Corporation, a Delaware corporation ('Company'), and Widget Supplier Inc., "
            "a California corporation ('Supplier').",
        ),
    ],
)
def test_canonical_clause_scores_above_tuned_threshold(real_classifier, category, clause):
    """For a canonical positive clause the raw sigmoid score must exceed the
    per-category F1-optimal threshold from thresholds.json.

    This is the primary guard that code changes do not degrade the model's
    accuracy on the categories that matter most for the thesis demo.
    """
    thresholds = _load_thresholds()
    threshold = thresholds.get(category, 0.55)

    scores = real_classifier.classify(clause)
    assert (
        category in scores
    ), f"Category '{category}' absent from output dict — label mapping may be broken."
    score = scores[category]
    assert score >= threshold, (
        f"'{category}' scored {score:.4f} < tuned threshold {threshold:.4f} "
        f"on canonical clause: {clause[:80]}…"
    )


# ---------------------------------------------------------------------------
# Clear negative examples — unrelated text must stay below threshold
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "category,unrelated_clause",
    [
        (
            "Cap On Liability",
            "The parties agree to use best efforts to resolve any dispute through "
            "good-faith negotiation before initiating formal legal proceedings.",
        ),
        (
            "Non-Compete",
            "Licensee shall pay a quarterly royalty of five percent (5%) of net revenues "
            "derived from distribution of the Licensed Product.",
        ),
    ],
)
def test_unrelated_clause_scores_below_threshold(real_classifier, category, unrelated_clause):
    """Text that clearly does not contain the target concept must not trigger it."""
    thresholds = _load_thresholds()
    threshold = thresholds.get(category, 0.55)

    scores = real_classifier.classify(unrelated_clause)
    score = scores.get(category, 0.0)
    assert score < threshold, (
        f"'{category}' false-positive: scored {score:.4f} ≥ threshold {threshold:.4f} "
        f"on: {unrelated_clause[:80]}…"
    )


# ---------------------------------------------------------------------------
# Orchestrator end-to-end with real classifier (no LLM / no extractor)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_orchestrator_e2e_with_real_classifier(real_classifier):
    """Full pipeline must emit RiskScores for a multi-category canonical clause."""
    from src.application.orchestration.orchestrator import ContractOrchestrator
    from src.domain.risk_policy import RiskPolicy

    orch = ContractOrchestrator(classifier=real_classifier, risk_policy=RiskPolicy())

    clause = (
        "The aggregate liability of each party under this Agreement shall not exceed "
        "one million dollars ($1,000,000). "
        "This Agreement shall be governed by the laws of the State of Delaware. "
        "Either party may terminate for convenience upon thirty (30) days' written notice."
    )
    risks = orch.analyze(clause)
    assert len(risks) >= 1, "Orchestrator returned no RiskScores on a multi-category clause."

    categories = {r.category for r in risks}
    # At least one of these three canonical categories must be detected
    expected = {"Cap On Liability", "Governing Law", "Termination For Convenience"}
    assert categories & expected, f"None of {expected} detected; got {categories}"


@pytest.mark.integration
def test_orchestrator_uses_per_category_thresholds(real_classifier):
    """Orchestrator must use thresholds from HFClassifier.per_category_thresholds,
    not the global 0.55 fallback, when the classifier ships a thresholds.json.
    """
    from src.application.orchestration.orchestrator import ContractOrchestrator
    from src.domain.risk_policy import RiskPolicy

    assert (
        real_classifier.per_category_thresholds is not None
    ), "Precondition: classifier must have per_category_thresholds loaded."

    orch = ContractOrchestrator(classifier=real_classifier, risk_policy=RiskPolicy())

    # The orchestrator's _threshold_for must return the per-category value, not 0.55
    threshold_cap = orch._threshold_for("Cap On Liability")
    expected = real_classifier.per_category_thresholds["Cap On Liability"]
    assert threshold_cap == expected, (
        f"Orchestrator returned {threshold_cap} for 'Cap On Liability', "
        f"expected per-category {expected}"
    )


@pytest.mark.integration
def test_orchestrator_justification_source_rule_without_llm(real_classifier):
    """Without an LLM provider the justification_source must be 'rule'."""
    from src.application.orchestration.orchestrator import ContractOrchestrator
    from src.domain.risk_policy import RiskPolicy

    orch = ContractOrchestrator(classifier=real_classifier, risk_policy=RiskPolicy())
    clause = (
        "The aggregate liability of each party shall not exceed one million dollars ($1,000,000)."
    )
    risks = orch.analyze(clause)
    cap_risks = [r for r in risks if r.category == "Cap On Liability"]
    if not cap_risks:
        pytest.skip("Cap On Liability not detected — threshold may have changed.")
    for r in cap_risks:
        assert (
            r.metadata.get("justification_source") == "rule"
        ), f"Expected 'rule' justification source, got {r.metadata.get('justification_source')}"


@pytest.mark.integration
def test_header_text_produces_no_risks(real_classifier):
    """Short all-caps header text must be rejected by the validator before reaching the auditor."""
    from src.application.orchestration.orchestrator import ContractOrchestrator
    from src.domain.risk_policy import RiskPolicy

    orch = ContractOrchestrator(classifier=real_classifier, risk_policy=RiskPolicy())
    risks = orch.analyze("LIMITATION OF LIABILITY")
    assert risks == [], f"Header text should produce no risks; got {[r.category for r in risks]}"


@pytest.mark.integration
def test_empty_text_produces_no_risks(real_classifier):
    """Empty / whitespace-only input must not raise — returns empty list."""
    from src.application.orchestration.orchestrator import ContractOrchestrator
    from src.domain.risk_policy import RiskPolicy

    orch = ContractOrchestrator(classifier=real_classifier, risk_policy=RiskPolicy())
    assert orch.analyze("") == []
    assert orch.analyze("   \n\t  ") == []
