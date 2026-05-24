"""
Background worker that turns an uploaded contract bytes blob into a job result.

Runs in a ThreadPoolExecutor (NOT asyncio) because the orchestrator and
DocumentNormalizer are blocking CPU-bound calls. The FastAPI event loop
stays free for /jobs polling.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from src.api.jobs import JobStore
from src.application.orchestration.orchestrator import ContractOrchestrator
from src.data.document_normalizer import DocumentFormat, DocumentNormalizerFactory
from src.data.sliding_window import SlidingWindowTokenizer, SlidingWindowTokenizerConfig

logger = logging.getLogger(__name__)


def _risk_to_dict(risk) -> Dict[str, Any]:
    return {
        "category": risk.category,
        "risk_level": risk.risk_level,
        "score": risk.score,
        "justification": risk.justification,
        "extracted_span": risk.extracted_span,
        "metadata": risk.metadata,
        "span_start_offset": risk.span_start_offset,
        "span_end_offset": risk.span_end_offset,
        "source_doc": risk.source_doc,
    }


def run_analysis_job(
    *,
    job_id: str,
    file_bytes: bytes,
    filename: str,
    suffix: str,
    orchestrator: ContractOrchestrator,
    job_store: JobStore,
    window_size: int = 1500,
    overlap_size: int = 200,
    min_words_per_chunk: int = 20,
) -> None:
    """Parse, classify, and analyse a contract; persist the result in the job store.

    Any uncaught exception is recorded on the job as 'failed' with the
    error message; this is the only place we deliberately swallow
    exceptions, and we record them rather than mask them.
    """
    job_store.mark_running(job_id)
    try:
        # Persist bytes to a NamedTemporaryFile so the DocumentNormalizer
        # (which expects a Path) can use its existing format-detection
        # branch. delete=False because pypdf opens it lazily on Windows.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(file_bytes)
            tmp_path = Path(tf.name)

        try:
            normalized = DocumentNormalizerFactory.normalize(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        logger.info(
            "job %s parsed %s -> %d chars, %s pages",
            job_id,
            filename,
            len(normalized.content),
            normalized.page_count,
        )

        # Sliding-window analysis (same configuration as scripts/demo_e2e.py).
        tokenizer = SlidingWindowTokenizer(
            SlidingWindowTokenizerConfig(
                window_size=window_size,
                overlap_size=overlap_size,
            )
        )
        windows = tokenizer.tokenize(normalized.content, doc_filename=filename)

        seen_categories: set[str] = set()
        risks: List[Dict[str, Any]] = []
        for window in windows:
            if len(window.content.split()) < min_words_per_chunk:
                continue
            for risk in orchestrator.analyze(window.content, source_doc=filename):
                # Dedupe across overlapping windows: keep first sighting per category.
                if risk.category in seen_categories:
                    continue
                seen_categories.add(risk.category)
                risks.append(_risk_to_dict(risk))

        job_store.mark_completed(job_id, risks)
    except Exception as exc:
        # We do NOT mask the error — it's recorded verbatim on the job
        # row and surfaced via GET /api/v1/jobs/{id}.
        logger.exception("job %s failed during background analysis", job_id)
        job_store.mark_failed(job_id, f"{type(exc).__name__}: {exc}")


SUFFIX_TO_FORMAT = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".txt": DocumentFormat.TXT,
    ".md": DocumentFormat.MD,
}


def detect_format(filename: str) -> DocumentFormat:
    """Return DocumentFormat for a filename, raising ValueError on unsupported suffix."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUFFIX_TO_FORMAT:
        raise ValueError(
            f"Unsupported file extension {suffix!r}; " f"supported: {sorted(SUFFIX_TO_FORMAT)}"
        )
    return SUFFIX_TO_FORMAT[suffix]
