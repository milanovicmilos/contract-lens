"""Tests for the Contract domain entity."""

from datetime import datetime

from src.domain.contract import Clause, Contract, ContractMetadata
from src.domain.risk_score import RiskScore


def _risk(level: str, category: str = "Non-Compete") -> RiskScore:
    return RiskScore(
        category=category,
        risk_level=level,
        score=0.7,
        justification="test",
        extracted_span="span",
        metadata={},
    )


def test_contract_default_collections_are_empty():
    contract = Contract(
        raw_text="hello",
        metadata=ContractMetadata(source_path="a.txt", file_format="txt", char_count=5),
    )
    assert contract.clauses == []
    assert contract.risks == []


def test_add_risk_appends_to_list():
    contract = Contract(
        raw_text="hello",
        metadata=ContractMetadata(source_path="a.txt", file_format="txt", char_count=5),
    )
    contract.add_risk(_risk("High"))
    contract.add_risk(_risk("Low", category="License Grant"))
    assert len(contract.risks) == 2


def test_risk_summary_counts_by_level():
    contract = Contract(
        raw_text="hello",
        metadata=ContractMetadata(source_path="a.txt", file_format="txt", char_count=5),
    )
    for level in ["High", "High", "Medium", "Low"]:
        contract.add_risk(_risk(level))
    assert contract.risk_summary() == {"High": 2, "Medium": 1, "Low": 1}


def test_risks_by_level_filters_correctly():
    contract = Contract(
        raw_text="hello",
        metadata=ContractMetadata(source_path="a.txt", file_format="txt", char_count=5),
    )
    contract.add_risk(_risk("High", category="Non-Compete"))
    contract.add_risk(_risk("Low", category="Parties"))
    contract.add_risk(_risk("High", category="License Grant"))

    high = contract.risks_by_level("High")
    assert len(high) == 2
    assert {r.category for r in high} == {"Non-Compete", "License Grant"}


def test_categories_present_is_sorted_unique():
    contract = Contract(
        raw_text="hello",
        metadata=ContractMetadata(source_path="a.txt", file_format="txt", char_count=5),
    )
    contract.add_risk(_risk("High", category="Non-Compete"))
    contract.add_risk(_risk("High", category="Non-Compete"))
    contract.add_risk(_risk("Low", category="Parties"))
    assert contract.categories_present() == ["Non-Compete", "Parties"]


def test_metadata_holds_provenance():
    meta = ContractMetadata(
        source_path="/tmp/contract.pdf",
        file_format="pdf",
        title="Distribution Agreement",
        parties=["Acme Corp", "Beta LLC"],
        agreement_date="2024-01-15",
        char_count=10000,
        page_count=12,
        analyzed_at=datetime(2026, 5, 23, 12, 0, 0),
    )
    assert meta.title == "Distribution Agreement"
    assert meta.parties == ["Acme Corp", "Beta LLC"]
    assert meta.page_count == 12


def test_clause_offsets_round_trip():
    c = Clause(text="The Distributor shall not compete", start_offset=120, end_offset=153, page=2)
    assert c.end_offset - c.start_offset == 33
    assert c.page == 2
