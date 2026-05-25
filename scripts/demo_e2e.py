"""
End-to-end demo: parse a CUAD contract, classify, score risks, render PDF + JSON.

Usage:
    python scripts/demo_e2e.py CUAD_v1/full_contract_pdf/Part_I/.../something.pdf
    # or for a TXT/MD path:
    python scripts/demo_e2e.py path/to/contract.txt

Output:
    reports/<stem>_compliance.json
    reports/<stem>_compliance.pdf

The script exercises every Clean Architecture seam: DocumentNormalizerFactory
(infrastructure -> data), HFClassifier (infra -> domain via IClassifier),
RiskPolicy (domain), ContractOrchestrator (infra -> application), the
GenerateComplianceReport use case (application), and PDF rendering (infra).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.generate_compliance_report import GenerateComplianceReport  # noqa: E402
from src.application.orchestration.orchestrator import ContractOrchestrator  # noqa: E402
from src.data.document_normalizer import DocumentNormalizerFactory  # noqa: E402
from src.data.sliding_window import (  # noqa: E402
    SlidingWindowTokenizer,
    SlidingWindowTokenizerConfig,
)
from src.domain.contract import Contract, ContractMetadata  # noqa: E402
from src.domain.risk_policy import RiskPolicy  # noqa: E402
from src.infrastructure.ai.hf_classifier import HFClassifier  # noqa: E402
from src.infrastructure.database.chroma_wrapper import ChromaWrapper  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("demo_e2e")


def _build_llm_provider():
    """Try to build OpenAI provider; return None when no key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from src.infrastructure.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    except Exception as exc:
        logger.warning(f"Could not build LLM provider: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="End-to-end contract analysis demo")
    parser.add_argument("contract", type=Path, help="Path to a contract file (.pdf/.docx/.txt/.md)")
    parser.add_argument(
        "--classifier",
        default=os.getenv("CLASSIFIER_MODEL", "models/deberta-cuad-classifier"),
        help="Local fine-tuned classifier directory",
    )
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIR", "./chroma_db"),
        help="ChromaDB persist directory for RAG",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(os.getenv("REPORTS_DIR", "reports")),
        help="Where to write JSON/PDF reports",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1500,
        help="Sliding window character size for analysis",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip OpenAI calls even if an API key is present",
    )
    args = parser.parse_args()

    if not args.contract.exists():
        logger.error(f"Contract file not found: {args.contract}")
        sys.exit(1)

    # 1. Normalize document via Factory (auto-detects format from suffix)
    logger.info(f"Normalizing {args.contract.name}")
    normalized = DocumentNormalizerFactory.normalize(args.contract)
    logger.info(f"  -> {normalized.page_count} pages, {len(normalized.content)} chars")

    # 2. Build the orchestrator (classifier + policy + RAG + LLM)
    logger.info(f"Loading classifier from {args.classifier}")
    classifier = HFClassifier(model_name_or_path=args.classifier)
    vector_db = ChromaWrapper(persist_directory=args.chroma_dir, collection_name="legal_regulations")
    llm_provider = None if args.no_llm else _build_llm_provider()
    orchestrator = ContractOrchestrator(
        classifier=classifier,
        risk_policy=RiskPolicy(),
        extractor=None,
        llm_provider=llm_provider,
        vector_db=vector_db,
    )
    logger.info(f"LLM provider: {llm_provider.name() if llm_provider else '(none — local only)'}")

    # 3. Slide a window across the contract and analyse each chunk
    tok_cfg = SlidingWindowTokenizerConfig(
        window_size=args.window_size,
        overlap_size=200,
    )
    tokenizer = SlidingWindowTokenizer(tok_cfg)
    windows = tokenizer.tokenize(normalized.content, doc_filename=args.contract.stem)

    contract = Contract(
        raw_text=normalized.content,
        metadata=ContractMetadata(
            source_path=str(args.contract),
            file_format=normalized.format.value,
            title=normalized.metadata.get("title") or args.contract.stem,
            page_count=normalized.page_count,
            char_count=len(normalized.content),
            analyzed_at=datetime.now(timezone.utc),
        ),
    )

    logger.info(f"Analysing {len(windows)} sliding windows")
    seen_categories = set()
    for window in windows:
        text = window.content
        if len(text.split()) < 20:
            continue
        risks = orchestrator.analyze(text, source_doc=args.contract.name)
        for risk in risks:
            # Dedupe across overlapping windows: keep first sighting per category.
            if risk.category in seen_categories:
                continue
            seen_categories.add(risk.category)
            contract.add_risk(risk)

    logger.info(f"Detected {len(contract.risks)} unique risks: {contract.risk_summary()}")

    # 4. Emit compliance reports
    reporter = GenerateComplianceReport(output_dir=args.reports_dir)
    json_path = reporter.to_json(contract)
    pdf_path = reporter.to_pdf(contract)

    print(json_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
