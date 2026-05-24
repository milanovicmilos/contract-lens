"""
Domain value object for a single identified risk.

Lives in the domain layer because it is the canonical output of the risk
assessment process — every downstream layer (application use cases,
infrastructure persisters, API responses) ultimately depends on this shape.
Keeping it in `src/application/interfaces/` previously forced
`src/domain/contract.py` to import upward into the application layer, which
violated Clean Architecture's dependency rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RiskScore:
    """
    Represents the risk level and justification for a specific clause.

    Traceability fields (span_start_offset, span_end_offset, source_doc) enable
    auditing every risk back to the original contract text — a hard requirement
    from the project's Transparency goal: every Risk Score must be backed by a
    citation and a reference to the applicable text location.
    """

    category: str
    risk_level: str  # "Low", "Medium", "High"
    score: float  # 0.0 to 1.0 confidence or exact numerical score
    justification: str
    extracted_span: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Traceability (optional; default to None for backward compatibility with
    # callers that don't yet plumb offsets through).
    span_start_offset: Optional[int] = None
    span_end_offset: Optional[int] = None
    source_doc: Optional[str] = None
