"""
Application Use Case: Generate a Compliance Report.

Takes an analysed Contract aggregate and emits either:
- JSON: machine-readable, full fidelity (every RiskScore, traceability fields,
  metadata, summary counters). Used by the API and for downstream pipelines.
- PDF: human-readable summary used during legal review. Built with reportlab
  so it has no runtime dependency on a headless browser or LibreOffice.

The use case is pure orchestration — formatting concerns live in
src/infrastructure/reporting/, and we depend only on Domain types.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.contract import Contract

logger = logging.getLogger(__name__)


def _serialize_metadata(meta) -> Dict[str, Any]:
    d = asdict(meta)
    if d.get("analyzed_at") is not None:
        d["analyzed_at"] = d["analyzed_at"].isoformat()
    return d


class GenerateComplianceReport:
    """Render a `Contract` into a compliance report (JSON or PDF)."""

    def __init__(self, output_dir: Path = Path("reports")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- JSON
    def to_json(self, contract: Contract, output_name: Optional[str] = None) -> Path:
        name = output_name or self._default_name(contract, ".json")
        out_path = self.output_dir / name

        payload = {
            "metadata": _serialize_metadata(contract.metadata),
            "summary": contract.risk_summary(),
            "categories_present": contract.categories_present(),
            "risks": [
                {
                    "category": r.category,
                    "risk_level": r.risk_level,
                    "score": r.score,
                    "justification": r.justification,
                    "extracted_span": r.extracted_span,
                    "span_start_offset": r.span_start_offset,
                    "span_end_offset": r.span_end_offset,
                    "source_doc": r.source_doc,
                    "metadata": r.metadata,
                }
                for r in contract.risks
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote JSON compliance report -> {out_path}")
        return out_path

    # ------------------------------------------------------------------ PDF
    def to_pdf(self, contract: Contract, output_name: Optional[str] = None) -> Path:
        from src.infrastructure.reporting.pdf_renderer import render_contract_report

        name = output_name or self._default_name(contract, ".pdf")
        out_path = self.output_dir / name
        render_contract_report(contract, out_path)
        logger.info(f"Wrote PDF compliance report -> {out_path}")
        return out_path

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _default_name(contract: Contract, suffix: str) -> str:
        stem = Path(contract.metadata.source_path).stem or "contract"
        return f"{stem}_compliance{suffix}"
