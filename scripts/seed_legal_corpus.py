"""
Seed the local ChromaDB with the full legal corpus.

Replaces the original 12-snippet ``seed_regulations.py`` with a structured,
article-level corpus loaded from ``data/legal_corpus/*.jsonl``. Each row
is one GDPR article, one EU AI Act provision, or one practice principle —
the embedded text is paragraph-sized so RAG retrieval surfaces a precise
citation rather than a 50-page blob.

USAGE:
    python scripts/seed_legal_corpus.py \\
        --persist-dir ./chroma_db \\
        --collection legal_regulations

Idempotent: re-running upserts by stable id (e.g. "gdpr-art-28"), so it
is safe to re-seed after editing a corpus row.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script standalone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.legal_corpus import default_corpus_dir, group_by_source, load_corpus  # noqa: E402
from src.infrastructure.database.chroma_wrapper import ChromaWrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed(persist_dir: str, collection_name: str, corpus_dir: Path) -> int:
    docs = load_corpus(corpus_dir)
    by_source = group_by_source(docs)
    logger.info("Corpus breakdown: %s", by_source)

    db = ChromaWrapper(persist_directory=persist_dir, collection_name=collection_name)
    if db._collection is None:
        logger.error("ChromaDB collection unavailable; aborting.")
        return 1

    texts = [d.text for d in docs]
    metadatas = [d.to_metadata() for d in docs]
    ids = [d.id for d in docs]

    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info(
        "Seeded %d documents into '%s' at %s.", len(texts), collection_name, persist_dir
    )

    # Sanity-check: a query that should resolve to GDPR Art. 28.
    sample = db.search(
        "obligations of a processor when handling personal data on behalf of a controller",
        top_k=3,
    )
    logger.info("Sample retrieval (top %d):", len(sample))
    for i, r in enumerate(sample, 1):
        snippet = (r.get("text") or "")[:120].replace("\n", " ")
        logger.info("  %d. %s... (source=%s)", i, snippet, (r.get("metadata") or {}).get("source"))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ChromaDB with the legal corpus")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--collection", default="legal_regulations")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=default_corpus_dir(),
        help="Directory containing *.jsonl corpus files",
    )
    args = parser.parse_args()
    raise SystemExit(seed(args.persist_dir, args.collection, args.corpus_dir))


if __name__ == "__main__":
    main()
