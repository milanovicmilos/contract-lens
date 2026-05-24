"""
Background-job tracking for ContractLens contract uploads.

Backed by SQLite (stdlib) so jobs survive process restarts without
introducing a Redis / Celery dependency. Workers run in a
ThreadPoolExecutor because the orchestrator is CPU-bound (HF inference)
and FastAPI's event loop must stay responsive for /jobs polling.

Scope notes:
- Single-process only. Two API workers would race on the same SQLite
  file; a multi-worker deployment needs Postgres + a real queue
  (Celery / RQ). Documented in README under "Scaling beyond one node".
- No retention sweeper. Completed rows accumulate; a future PR will add
  a TTL-based cleaner if disk usage becomes an issue.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    api_key         TEXT,
    source_filename TEXT,
    source_format   TEXT,
    char_count      INTEGER,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT,
    result_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_api_key ON jobs (api_key);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs (status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """SQLite-backed job state store. Thread-safe via per-instance lock."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because workers (ThreadPoolExecutor) and
        # request handlers both write. The per-instance Lock serialises
        # writes; reads are short and acceptable under that lock.
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ writes
    def create(
        self,
        *,
        api_key: Optional[str],
        source_filename: str,
        source_format: str,
        char_count: int,
    ) -> str:
        job_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, api_key, source_filename, "
                "source_format, char_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    JobStatus.PENDING.value,
                    api_key,
                    source_filename,
                    source_format,
                    char_count,
                    _now(),
                ),
            )
            conn.commit()
        logger.info(
            "job %s created (file=%s, format=%s, chars=%d)",
            job_id,
            source_filename,
            source_format,
            char_count,
        )
        return job_id

    def mark_running(self, job_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                (JobStatus.RUNNING.value, _now(), job_id),
            )
            conn.commit()

    def mark_completed(self, job_id: str, result: List[Dict[str, Any]]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, result_json = ? " "WHERE id = ?",
                (
                    JobStatus.COMPLETED.value,
                    _now(),
                    json.dumps(result, ensure_ascii=False),
                    job_id,
                ),
            )
            conn.commit()
        logger.info("job %s completed (%d risks)", job_id, len(result))

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, error = ? " "WHERE id = ?",
                (JobStatus.FAILED.value, _now(), error[:2000], job_id),
            )
            conn.commit()
        logger.warning("job %s failed: %s", job_id, error[:200])

    # ------------------------------------------------------------------- reads
    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d: Dict[str, Any] = dict(row)
    if d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except json.JSONDecodeError:
            d["result"] = None
    else:
        d["result"] = None
    d.pop("result_json", None)
    return d


def default_db_path() -> Path:
    return Path(os.getenv("JOBS_DB_PATH", "data/jobs.sqlite"))
