"""Tests for the GenerateComplianceReport use case."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.application.generate_compliance_report import GenerateComplianceReport
from src.domain.contract import Contract, ContractMetadata
from src.domain.risk_score import RiskScore


def _sample_contract(source: str = "contract.txt") -> Contract:
    contract = Contract(
        raw_text="The Distributor shall not compete worldwide. Liability capped at $1000.",
        metadata=ContractMetadata(
            source_path=source,
            file_format="txt",
            title="Sample Agreement",
            parties=["Acme Corp", "Beta LLC"],
            governing_law="Delaware",
            char_count=72,
            analyzed_at=datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc),
        ),
    )
    contract.add_risk(
        RiskScore(
            category="Non-Compete",
            risk_level="High",
            score=0.9,
            justification="Worldwide non-compete restricts future business.",
            extracted_span="shall not compete worldwide",
            metadata={"classifier_confidence": 0.95},
            span_start_offset=20,
            span_end_offset=46,
            source_doc=source,
        )
    )
    contract.add_risk(
        RiskScore(
            category="Cap On Liability",
            risk_level="Medium",
            score=0.5,
            justification="Cap is reasonable but tight.",
            extracted_span="Liability capped at $1000",
            metadata={"classifier_confidence": 0.72},
            span_start_offset=48,
            span_end_offset=72,
            source_doc=source,
        )
    )
    return contract


def test_json_report_contains_summary_and_risks(tmp_path: Path):
    reporter = GenerateComplianceReport(output_dir=tmp_path)
    contract = _sample_contract()

    path = reporter.to_json(contract)
    assert path.exists()
    assert path.parent == tmp_path

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"] == {"High": 1, "Medium": 1, "Low": 0}
    assert payload["categories_present"] == ["Cap On Liability", "Non-Compete"]
    assert len(payload["risks"]) == 2

    first = next(r for r in payload["risks"] if r["category"] == "Non-Compete")
    assert first["risk_level"] == "High"
    assert first["span_start_offset"] == 20
    assert first["source_doc"] == "contract.txt"
    assert first["metadata"]["classifier_confidence"] == pytest.approx(0.95)


def test_json_report_default_filename_from_source(tmp_path: Path):
    reporter = GenerateComplianceReport(output_dir=tmp_path)
    path = reporter.to_json(_sample_contract(source="DistributionAgreement.pdf"))
    assert path.name == "DistributionAgreement_compliance.json"


def test_pdf_report_is_emitted(tmp_path: Path):
    reporter = GenerateComplianceReport(output_dir=tmp_path)
    path = reporter.to_pdf(_sample_contract())
    assert path.exists()
    assert path.suffix == ".pdf"
    # A non-trivial PDF must contain the %PDF header and an EOF marker.
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF")
    assert b"%%EOF" in raw[-1024:]


def test_pdf_report_handles_empty_risks(tmp_path: Path):
    reporter = GenerateComplianceReport(output_dir=tmp_path)
    empty_contract = Contract(
        raw_text="ARTICLE 1. DEFINITIONS",
        metadata=ContractMetadata(source_path="x.txt", file_format="txt", char_count=22),
    )
    path = reporter.to_pdf(empty_contract)
    assert path.exists()
    assert path.suffix == ".pdf"
