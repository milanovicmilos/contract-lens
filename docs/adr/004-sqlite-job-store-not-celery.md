# ADR-004 — SQLite-backed job store instead of Celery + Redis

**Status:** Accepted with a documented scaling cliff (2026-05-24, PR #6)
**Spec mapping:** Spec §3 (Application Layer) implicitly assumes a
single-deployment unit; multi-worker production not in scope.

## Context

The original `POST /api/v1/analyze` endpoint was synchronous and
accepted a `text: str` body. For any contract longer than a few
clauses this hits the HTTP timeout — parse + classify + per-chunk LLM
calls easily blow past 30 s for a 50-page PDF.

PR #6 added an upload-and-poll flow:
- `POST /api/v1/contracts` (multipart) → 202 with a job id.
- Background worker parses, runs the orchestrator over sliding windows,
  persists results.
- `GET /api/v1/jobs/{id}` → status + result when ready.

The job store had to satisfy:
1. Survive an API restart (PoC users rebuild containers regularly).
2. Be visible across the request handler and the background worker (in
   different threads).
3. Not introduce an infra dependency that gates dev / CI runs.
4. Be auditable (which key submitted which job, when, with what file).

## Decision

Use **SQLite via the stdlib `sqlite3` module** as the job store, with
a single `JobStore` class wrapping all writes behind a per-instance
`threading.Lock`. Workers run in a **`ThreadPoolExecutor`**
(`JOB_WORKERS`, default 2) created at FastAPI lifespan startup and
shut down at lifespan teardown.

Schema is one table:

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,      -- pending | running | completed | failed
    api_key TEXT,              -- which key submitted (for cross-tenant 404)
    source_filename TEXT,
    source_format TEXT,
    char_count INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,                -- exception text on failure (no masking)
    result_json TEXT           -- serialised RiskScore list on success
);
```

Cross-tenant access returns **404 (not 403)** to avoid leaking
job-existence information.

## Consequences

**Wins:**
- Zero new infra dependencies. Dev, CI, and Docker images all run
  unchanged. SQLite is in the stdlib.
- 6 integration tests in `tests/test_api_upload.py` drive the **full
  path** (real ThreadPoolExecutor + real SQLite + mocked orchestrator)
  in ~10 s. No mocking of the persistence layer means we test what we
  ship.
- Jobs survive a container restart. Operators can re-attach to a
  running analysis after a deploy if they kept the volume.
- Error propagation is honest: any exception in `run_analysis_job` is
  recorded verbatim in the `error` column and surfaced via GET — we
  do not swallow it.

**Costs — and the documented scaling cliff:**
- **Single-process only.** Two uvicorn workers would race on the
  SQLite write lock. Documented in source
  (`src/api/jobs.py` module docstring) and in `README.md` Endpoints
  section: "Single-process only — scaling to multiple API workers
  needs Postgres + Celery/RQ."
- No retention sweeper. Completed rows accumulate. A TTL cleaner is
  flagged as a follow-up TODO; current operating volume (one user
  during defence) means disk usage is irrelevant.
- ThreadPoolExecutor caps parallelism per worker. CPU-bound HF
  inference is the bottleneck anyway; adding more threads would not
  help past ~2 because the GIL holds the orchestrator's tokeniser
  call serially.

## Alternatives considered

1. **Celery + Redis + a results backend.** Right answer at scale,
   wrong answer at this stage of the project. Adds two new processes,
   two new env-vars (broker, backend), one new container, and a deploy
   runbook. The thesis defence environment doesn't justify it.
2. **In-memory dict + threading.Lock.** Considered. Rejected because
   jobs would not survive a container restart, which is the most
   common operator action in dev. SQLite is "one file on disk" — same
   simplicity, persistence for free.
3. **Postgres on the dev machine.** Same problems as Celery for the
   current scope: an external process to manage. A future
   migration is straightforward — the `JobStore` class is the only
   place that constructs SQL strings; swapping its driver in a follow-up
   PR is a contained change.
4. **FastAPI BackgroundTasks alone, no executor.** Rejected because
   BackgroundTasks run on the event loop; a blocking HF inference
   call would freeze the API for the duration of the analysis.

## References

- PR #6 (`feat(api): async PDF/DOCX upload + SQLite job store + jobs polling`).
- `src/api/jobs.py` — single source for the JobStore implementation.
- The "Scaling cliff" callout in `README.md` "Endpoints" section.
