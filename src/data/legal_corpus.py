"""
Legal corpus loader for the RAG store.

The corpus lives as JSONL files under ``data/legal_corpus/`` so the data
is version-controlled, auditable, and trivial to extend without touching
code. Each row is one article (GDPR), one provision (EU AI Act), or one
practice principle. Article-level granularity keeps the RAG matches
precise — when the Consultant agent queries "data transfer to third
country", retrieval returns the GDPR Chapter V article, not a 50-page
blob.

Sources and license
-------------------
EU primary legislation (GDPR — Regulation (EU) 2016/679, EU AI Act —
Regulation (EU) 2024/1689) is freely reusable under Commission Decision
2011/833/EU. The texts under data/legal_corpus/ are condensed,
paraphrased article summaries — not verbatim reproductions — keyed by
the official article number so a reviewer can verify against EUR-Lex.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegalDocument:
    """One row from the corpus — used both for seeding and as a retrieval result."""

    id: str  # stable, idempotent — e.g. "gdpr-art-28"
    source: str  # "GDPR", "EU AI Act", "Practice Note", ...
    title: str
    text: str  # what gets embedded
    article: Optional[str] = None  # e.g. "28" for GDPR, "5" for AI Act
    topic: Optional[str] = None  # human-readable grouping (e.g. "data-transfers")
    tags: tuple = ()  # additional facets for filtered retrieval

    def to_metadata(self) -> Dict[str, Any]:
        """Chroma metadata payload (only scalars / strings allowed)."""
        meta: Dict[str, Any] = {"source": self.source, "title": self.title}
        if self.article:
            meta["article"] = self.article
        if self.topic:
            meta["topic"] = self.topic
        if self.tags:
            meta["tags"] = ",".join(self.tags)
        return meta


def default_corpus_dir() -> Path:
    """data/legal_corpus/ relative to the repo root."""
    return Path(__file__).resolve().parents[2] / "data" / "legal_corpus"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: malformed JSON ({exc.msg})") from exc


_REQUIRED_FIELDS = ("id", "source", "title", "text")


def _row_to_doc(row: Dict[str, Any], source_path: Path) -> LegalDocument:
    missing = [k for k in _REQUIRED_FIELDS if not row.get(k)]
    if missing:
        raise ValueError(
            f"{source_path.name}: row id={row.get('id')!r} missing required fields: {missing}"
        )
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return LegalDocument(
        id=str(row["id"]),
        source=str(row["source"]),
        title=str(row["title"]),
        text=str(row["text"]).strip(),
        article=str(row["article"]) if row.get("article") is not None else None,
        topic=str(row["topic"]) if row.get("topic") else None,
        tags=tuple(tags),
    )


def load_corpus(corpus_dir: Optional[Path] = None) -> List[LegalDocument]:
    """Load every ``*.jsonl`` file under ``corpus_dir`` and return all articles.

    Raises FileNotFoundError if the directory is missing or empty — silent
    fallback to an empty corpus would hide a real deployment misconfiguration.
    """
    corpus_dir = Path(corpus_dir or default_corpus_dir())
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Legal corpus directory not found: {corpus_dir}. "
            "Run scripts/seed_legal_corpus.py or check the JOBS_DB_PATH env."
        )

    docs: List[LegalDocument] = []
    seen_ids: set[str] = set()
    files = sorted(corpus_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files found in legal corpus directory: {corpus_dir}")

    for path in files:
        file_doc_count = 0
        for row in _iter_jsonl(path):
            doc = _row_to_doc(row, path)
            if doc.id in seen_ids:
                raise ValueError(f"Duplicate corpus id {doc.id!r} (second sighting in {path.name})")
            seen_ids.add(doc.id)
            docs.append(doc)
            file_doc_count += 1
        logger.info("Loaded %d documents from %s", file_doc_count, path.name)

    logger.info("Legal corpus loaded: %d documents from %d files.", len(docs), len(files))
    return docs


def group_by_source(docs: Iterable[LegalDocument]) -> Dict[str, int]:
    """Counts per source, useful for the seed script's summary log."""
    counts: Dict[str, int] = {}
    for d in docs:
        counts[d.source] = counts.get(d.source, 0) + 1
    return counts
