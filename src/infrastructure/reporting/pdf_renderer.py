"""
PDF rendering of compliance reports using reportlab.

The renderer is intentionally lightweight: a title page with the contract
metadata, a colour-coded risk summary box, and one paragraph per RiskScore
showing category, level, citation and justification. No external assets,
no headless browser, no LibreOffice — works in any Python environment that
has reportlab installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.domain.contract import Contract
from src.domain.risk_score import RiskScore

RISK_COLOURS = {
    "High": colors.HexColor("#dc2626"),
    "Medium": colors.HexColor("#f59e0b"),
    "Low": colors.HexColor("#16a34a"),
}


def _build_styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="RiskHigh",
            parent=base["Normal"],
            textColor=colors.HexColor("#dc2626"),
            spaceAfter=4,
            alignment=TA_LEFT,
        )
    )
    base.add(
        ParagraphStyle(
            name="RiskMedium",
            parent=base["Normal"],
            textColor=colors.HexColor("#b45309"),
            spaceAfter=4,
            alignment=TA_LEFT,
        )
    )
    base.add(
        ParagraphStyle(
            name="RiskLow",
            parent=base["Normal"],
            textColor=colors.HexColor("#15803d"),
            spaceAfter=4,
            alignment=TA_LEFT,
        )
    )
    base.add(
        ParagraphStyle(
            name="Justification",
            parent=base["Normal"],
            fontSize=9,
            leftIndent=14,
            textColor=colors.HexColor("#374151"),
            spaceAfter=10,
        )
    )
    return base


def _summary_table(contract: Contract) -> Table:
    summary = contract.risk_summary()
    data = [
        ["Risk Level", "Count"],
        ["High", str(summary.get("High", 0))],
        ["Medium", str(summary.get("Medium", 0))],
        ["Low", str(summary.get("Low", 0))],
        ["Total Risks", str(len(contract.risks))],
        ["Categories Detected", str(len(contract.categories_present()))],
    ]
    table = Table(data, colWidths=[180, 80])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("TEXTCOLOR", (0, 1), (0, 1), RISK_COLOURS["High"]),
                ("TEXTCOLOR", (0, 2), (0, 2), RISK_COLOURS["Medium"]),
                ("TEXTCOLOR", (0, 3), (0, 3), RISK_COLOURS["Low"]),
            ]
        )
    )
    return table


def _format_risk(risk: RiskScore, styles) -> List[Paragraph]:
    badge_style = (
        styles[f"Risk{risk.risk_level}"] if f"Risk{risk.risk_level}" in styles else styles["Normal"]
    )
    citation = ""
    if risk.span_start_offset is not None and risk.span_end_offset is not None:
        citation = f" (offset {risk.span_start_offset}-{risk.span_end_offset})"
    source = f" — {risk.source_doc}" if risk.source_doc else ""
    header_text = f"<b>[{risk.risk_level}] {risk.category}</b>{citation}{source}"
    return [
        Paragraph(header_text, badge_style),
        Paragraph(f"<i>Justification:</i> {risk.justification}", styles["Justification"]),
        Paragraph(
            f'<font color="#6b7280">"{risk.extracted_span[:400]}"</font>',
            styles["Justification"],
        ),
    ]


def render_contract_report(contract: Contract, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        title="ContractLens Compliance Report",
        author="ContractLens",
    )
    styles = _build_styles()
    story: List = []

    title = contract.metadata.title or "Contract Compliance Report"
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 10))

    meta_lines = [
        f"<b>Source:</b> {contract.metadata.source_path}",
        f"<b>Format:</b> {contract.metadata.file_format.upper()}",
        f"<b>Length:</b> {contract.metadata.char_count} chars",
    ]
    if contract.metadata.governing_law:
        meta_lines.append(f"<b>Governing law:</b> {contract.metadata.governing_law}")
    if contract.metadata.parties:
        meta_lines.append(f"<b>Parties:</b> {', '.join(contract.metadata.parties)}")
    if contract.metadata.analyzed_at:
        meta_lines.append(f"<b>Analyzed at:</b> {contract.metadata.analyzed_at.isoformat()}")
    for line in meta_lines:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Risk Summary</b>", styles["Heading2"]))
    story.append(_summary_table(contract))
    story.append(Spacer(1, 16))

    if contract.risks:
        story.append(Paragraph("<b>Detected Risks</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))
        ordered = sorted(
            contract.risks,
            key=lambda r: ({"High": 0, "Medium": 1, "Low": 2}.get(r.risk_level, 3), -r.score),
        )
        for risk in ordered:
            story.extend(_format_risk(risk, styles))
    else:
        story.append(Paragraph("<i>No risks detected by the policy engine.</i>", styles["Normal"]))

    doc.build(story)
    return output_path
