"""
Seed the local ChromaDB with regulatory snippets used by the Legal Consultant
agent's RAG step.

The snippets below are paraphrased from publicly available primary sources
(GDPR text, EU AI Act, contract law principles) and serve as starter context.
For production use, replace this curated set with a full regulation corpus.

USAGE:
    python scripts/seed_regulations.py \
        --persist-dir ./chroma_db \
        --collection legal_regulations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Allow running this script standalone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infrastructure.database.chroma_wrapper import ChromaWrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Each snippet is keyed by a stable ID for idempotent re-seeding. The 'tags'
# field is for filtering downstream (not yet used by the consultant, but
# captured so the corpus can be sliced by topic later).
REGULATION_SNIPPETS: List[Dict] = [
    # ----- GDPR (Regulation (EU) 2016/679) -----
    {
        "id": "gdpr-art-28-dpa",
        "text": (
            "GDPR Article 28 — Processor obligations. Processing by a processor shall be governed "
            "by a contract that sets out the subject matter, duration, nature and purpose of "
            "the processing, the type of personal data, the categories of data subjects, and the "
            "obligations and rights of the controller. The contract must require the processor to "
            "process personal data only on documented instructions, ensure confidentiality, "
            "implement security measures, engage sub-processors only with prior authorisation, "
            "and assist the controller with data subject requests."
        ),
        "metadata": {"source": "GDPR", "article": "28", "tags": "dpa,processor,subprocessor"},
    },
    {
        "id": "gdpr-art-32-security",
        "text": (
            "GDPR Article 32 — Security of processing. Controllers and processors shall implement "
            "appropriate technical and organisational measures to ensure a level of security "
            "appropriate to the risk, including pseudonymisation and encryption, ongoing "
            "confidentiality, integrity, availability and resilience of processing systems, "
            "and the ability to restore access in a timely manner."
        ),
        "metadata": {"source": "GDPR", "article": "32", "tags": "security,encryption"},
    },
    {
        "id": "gdpr-art-83-fines",
        "text": (
            "GDPR Article 83 — Administrative fines. Infringements of obligations under Articles "
            "8, 11, 25 to 39, 42 and 43 are subject to fines up to EUR 10 million or 2% of "
            "worldwide annual turnover, whichever is higher. Infringements of basic processing "
            "principles, data subject rights, or international transfer rules are subject to "
            "fines up to EUR 20 million or 4% of worldwide annual turnover."
        ),
        "metadata": {"source": "GDPR", "article": "83", "tags": "fines,penalties"},
    },
    {
        "id": "gdpr-art-44-transfers",
        "text": (
            "GDPR Chapter V — International transfers. Personal data may only be transferred to "
            "third countries if the Commission has decided that country ensures an adequate level "
            "of protection, or if appropriate safeguards (Standard Contractual Clauses, Binding "
            "Corporate Rules) are in place, or if a specific derogation applies. Transfers without "
            "such safeguards are unlawful and trigger Article 83 fines."
        ),
        "metadata": {"source": "GDPR", "chapter": "V", "tags": "transfers,scc,bcr"},
    },

    # ----- EU AI Act (Regulation (EU) 2024/1689) -----
    {
        "id": "ai-act-art-5-prohibited",
        "text": (
            "EU AI Act Article 5 — Prohibited AI practices. The placing on the market, putting "
            "into service, or use of AI systems that deploy subliminal techniques to materially "
            "distort behaviour, exploit vulnerabilities of specific groups, perform social scoring "
            "for general use, or conduct real-time remote biometric identification in public "
            "spaces (with narrow law-enforcement exceptions) is prohibited."
        ),
        "metadata": {"source": "EU AI Act", "article": "5", "tags": "prohibited,biometric"},
    },
    {
        "id": "ai-act-art-6-high-risk",
        "text": (
            "EU AI Act Article 6 — High-risk AI systems. AI systems intended for use as safety "
            "components in critical infrastructure, education, employment, essential services, "
            "law enforcement, migration, justice administration, or biometric identification are "
            "classified as high-risk and must comply with risk management, data governance, "
            "transparency, human oversight, and accuracy requirements."
        ),
        "metadata": {"source": "EU AI Act", "article": "6", "tags": "high-risk,compliance"},
    },
    {
        "id": "ai-act-art-99-penalties",
        "text": (
            "EU AI Act Article 99 — Penalties. Non-compliance with prohibited AI practices is "
            "subject to fines up to EUR 35 million or 7% of worldwide annual turnover. "
            "Non-compliance with most other obligations is subject to fines up to EUR 15 million "
            "or 3% of worldwide annual turnover."
        ),
        "metadata": {"source": "EU AI Act", "article": "99", "tags": "penalties,fines"},
    },

    # ----- Common-law contract principles (paraphrased for RAG context) -----
    {
        "id": "principle-uncapped-liability",
        "text": (
            "Contract risk principle — Unlimited liability. Clauses that impose unlimited or "
            "uncapped liability on a party create significant financial exposure. Standard "
            "practice limits liability to either the contract value, the fees paid in the prior "
            "12 months, or a fixed cap. Carve-outs for gross negligence, willful misconduct, "
            "indemnification obligations, and confidentiality breaches are common."
        ),
        "metadata": {"source": "Practice Note", "topic": "Liability", "tags": "liability,cap"},
    },
    {
        "id": "principle-ip-assignment",
        "text": (
            "Contract risk principle — IP assignment. 'Work made for hire' or full IP assignment "
            "clauses transfer ownership permanently. Before signing, verify carve-outs for "
            "pre-existing IP, background IP, and tools/methodologies developed independently. "
            "Joint ownership creates licensing complexity and should be avoided unless intended."
        ),
        "metadata": {"source": "Practice Note", "topic": "IP", "tags": "ip,assignment,wfh"},
    },
    {
        "id": "principle-non-compete",
        "text": (
            "Contract risk principle — Non-compete clauses. Reasonableness factors include "
            "geographic scope (worldwide bans are rarely enforceable), duration (typically 6-24 "
            "months), and the legitimate business interest protected. Many jurisdictions (e.g., "
            "California, EU member states) limit or void overbroad non-competes. The US FTC's "
            "2024 rule attempted a nationwide ban on most worker non-competes."
        ),
        "metadata": {"source": "Practice Note", "topic": "Non-Compete", "tags": "non-compete"},
    },
    {
        "id": "principle-change-of-control",
        "text": (
            "Contract risk principle — Change of control. CoC clauses that trigger automatic "
            "termination on M&A events can derail deals. Acceptable variants require notice and "
            "consent rather than automatic termination, or exclude internal reorganisations and "
            "affiliate transfers from the trigger."
        ),
        "metadata": {"source": "Practice Note", "topic": "M&A", "tags": "coc,m&a"},
    },
    {
        "id": "principle-governing-law-sanctions",
        "text": (
            "Contract risk principle — Sanctioned jurisdictions. Governing law clauses pointing "
            "to OFAC-sanctioned countries (e.g., Russia, Iran, North Korea, Cuba, Syria, regions "
            "of Ukraine) create enforcement risk and potential secondary-sanction exposure for "
            "third parties. Standard alternatives: English law, Delaware law, Singapore law."
        ),
        "metadata": {"source": "Practice Note", "topic": "Jurisdiction", "tags": "sanctions,ofac"},
    },
]


def seed(persist_dir: str, collection_name: str) -> int:
    """Idempotently add the curated snippet set to the configured collection."""
    db = ChromaWrapper(persist_directory=persist_dir, collection_name=collection_name)
    if db._collection is None:
        logger.error("ChromaDB collection unavailable; aborting.")
        return 1

    texts = [s["text"] for s in REGULATION_SNIPPETS]
    metadatas = [s["metadata"] for s in REGULATION_SNIPPETS]
    ids = [s["id"] for s in REGULATION_SNIPPETS]

    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info(f"Seeded {len(texts)} snippets into '{collection_name}' at {persist_dir}")

    sample_q = "What are the obligations for processing personal data under GDPR?"
    results = db.search(sample_q, top_k=3)
    logger.info(f"Sanity search '{sample_q}' returned {len(results)} results:")
    for i, r in enumerate(results, 1):
        snippet = r.get("text", "")[:80].replace("\n", " ")
        logger.info(f"  {i}. {snippet}... (score={r.get('score', 0):.3f})")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Seed ChromaDB with regulatory snippets")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--collection", default="legal_regulations")
    args = parser.parse_args()
    raise SystemExit(seed(args.persist_dir, args.collection))


if __name__ == "__main__":
    main()
