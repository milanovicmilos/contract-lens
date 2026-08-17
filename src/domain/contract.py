"""
Contract domain entity.

Pure-Python value objects with no dependencies on infrastructure (no HF, no
LangChain, no FastAPI). Used by application use cases as the canonical
representation of a parsed legal document and its analysed risk surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.domain.risk_score import RiskScore


@dataclass
class Clause:
    """A single text span identified inside a Contract."""

    text: str
    start_offset: int
    end_offset: int
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class ContractMetadata:
    """Bibliographic / provenance metadata recovered from the source document."""

    source_path: str
    file_format: str  # 'pdf', 'docx', 'txt', 'md'
    title: Optional[str] = None
    parties: List[str] = field(default_factory=list)
    agreement_date: Optional[str] = None
    governing_law: Optional[str] = None
    page_count: Optional[int] = None
    char_count: int = 0
    analyzed_at: Optional[datetime] = None
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class Contract:
    """Aggregate root for a contract under analysis."""

    raw_text: str
    metadata: ContractMetadata
    clauses: List[Clause] = field(default_factory=list)
    risks: List[RiskScore] = field(default_factory=list)

    def add_risk(self, risk: RiskScore) -> None:
        self.risks.append(risk)

    def risks_by_level(self, level: str) -> List[RiskScore]:
        return [r for r in self.risks if r.risk_level == level]

    def risk_summary(self) -> Dict[str, int]:
        summary = {"High": 0, "Medium": 0, "Low": 0}
        for r in self.risks:
            summary[r.risk_level] = summary.get(r.risk_level, 0) + 1
        return summary

    def categories_present(self) -> List[str]:
        return sorted({r.category for r in self.risks})
