"""Tests for the legal corpus loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.legal_corpus import (
    LegalDocument,
    default_corpus_dir,
    group_by_source,
    load_corpus,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_corpus_returns_typed_documents(tmp_path):
    """Every JSON row should round-trip into a LegalDocument with metadata."""
    _write_jsonl(
        tmp_path / "gdpr.jsonl",
        [
            {
                "id": "gdpr-art-28",
                "source": "GDPR",
                "article": "28",
                "title": "Processor",
                "topic": "processor",
                "tags": ["dpa", "subprocessor"],
                "text": "Processing by a processor shall be governed by a contract...",
            }
        ],
    )

    docs = load_corpus(tmp_path)

    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, LegalDocument)
    assert d.id == "gdpr-art-28"
    assert d.source == "GDPR"
    assert d.article == "28"
    assert "dpa" in d.tags
    meta = d.to_metadata()
    assert meta["source"] == "GDPR"
    assert meta["article"] == "28"
    assert "dpa" in meta["tags"]  # comma-joined


def test_load_corpus_merges_multiple_files(tmp_path):
    _write_jsonl(
        tmp_path / "gdpr.jsonl",
        [{"id": "gdpr-1", "source": "GDPR", "title": "A", "text": "Body A."}],
    )
    _write_jsonl(
        tmp_path / "ai_act.jsonl",
        [{"id": "ai-1", "source": "EU AI Act", "title": "B", "text": "Body B."}],
    )

    docs = load_corpus(tmp_path)
    assert {d.id for d in docs} == {"gdpr-1", "ai-1"}
    assert group_by_source(docs) == {"GDPR": 1, "EU AI Act": 1}


def test_load_corpus_rejects_duplicate_ids(tmp_path):
    _write_jsonl(
        tmp_path / "a.jsonl",
        [{"id": "dup", "source": "GDPR", "title": "A", "text": "A"}],
    )
    _write_jsonl(
        tmp_path / "b.jsonl",
        [{"id": "dup", "source": "GDPR", "title": "B", "text": "B"}],
    )
    with pytest.raises(ValueError, match="Duplicate corpus id 'dup'"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_missing_required_field(tmp_path):
    _write_jsonl(
        tmp_path / "broken.jsonl",
        [{"id": "x", "source": "GDPR", "title": "T"}],  # no text
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_malformed_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "source": "GDPR"', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_empty_directory(tmp_path):
    """An empty directory must fail loudly — silent empty corpus would mask a real issue."""
    with pytest.raises(FileNotFoundError, match="No \\*.jsonl files"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Legal corpus directory not found"):
        load_corpus(tmp_path / "does-not-exist")


def test_default_corpus_dir_points_at_repo_data():
    """default_corpus_dir resolves to repo_root/data/legal_corpus."""
    d = default_corpus_dir()
    assert d.name == "legal_corpus"
    assert d.parent.name == "data"


def test_repo_corpus_loads_cleanly():
    """The corpus shipped in data/legal_corpus/ must parse without errors."""
    docs = load_corpus()
    assert len(docs) >= 40, "expected at least 40 articles across GDPR + AI Act + practice"
    sources = group_by_source(docs)
    assert "GDPR" in sources
    assert "EU AI Act" in sources
    assert "Practice Note" in sources
