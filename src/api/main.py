"""
FastAPI entry point for ContractLens.

Startup is lenient: missing OPENAI_API_KEY only disables the Legal Consultant
LLM call (the rest of the pipeline still works in local-only mode). Model
selection and provider wiring is configurable via env vars so production can
override the dev defaults (small models for cold-start speed in CI / local dev).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.jobs import JobStatus, JobStore, default_db_path
from src.api.logging_config import configure_logging
from src.api.metrics import MetricsMiddleware, metrics_enabled, render_metrics
from src.api.middleware import BodySizeLimitMiddleware, RequestIDMiddleware
from src.api.rate_limit import limiter
from src.api.security import require_api_key, warn_if_insecure
from src.api.upload_worker import detect_format, run_analysis_job
from src.application.generate_compliance_report import GenerateComplianceReport
from src.application.interfaces.illm_provider import ILLMProvider
from src.application.orchestration.orchestrator import ContractOrchestrator
from src.domain.contract import Contract, ContractMetadata
from src.domain.risk_policy import RiskPolicy
from src.infrastructure.ai.hf_classifier import HFClassifier
from src.infrastructure.database.chroma_wrapper import ChromaWrapper

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (env-driven so callers can override without code changes)
# ---------------------------------------------------------------------------
def _is_truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "prajjwal1/bert-tiny")
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_regulations")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DISABLE_LLM = _is_truthy(os.getenv("DISABLE_LLM"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))
# CORS allowlist — comma-separated origins, empty disables CORS entirely
# (most restrictive default). Use ["*"] in dev only via ALLOWED_ORIGINS="*".
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]


orchestrator: Optional[ContractOrchestrator] = None
report_generator: Optional[GenerateComplianceReport] = None
job_store: Optional[JobStore] = None
job_executor: Optional[ThreadPoolExecutor] = None

# Worker pool size — single-process default, override via env in production.
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "2")))


def _build_llm_provider() -> Optional[ILLMProvider]:
    """Try to construct an ILLMProvider (OpenAI by default); return None on any failure."""
    if DISABLE_LLM:
        logger.info("LLM disabled via DISABLE_LLM env var.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set; Legal Consultant will fall back to rule-based notes."
        )
        return None

    try:
        from src.infrastructure.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model=OPENAI_MODEL, api_key=api_key)
    except Exception as exc:
        logger.error(f"Failed to initialize OpenAIProvider: {exc}; running without LLM.")
        return None


def _build_extractor():
    """Optionally build the DeBERTa extractor if EXTRACTOR_MODEL points to a model."""
    if not EXTRACTOR_MODEL:
        return None
    try:
        from src.infrastructure.ai.deberta_extractor import DebertaExtractor

        return DebertaExtractor(model_name_or_path=EXTRACTOR_MODEL)
    except Exception as exc:
        logger.error(f"Failed to load extractor '{EXTRACTOR_MODEL}': {exc}")
        return None


def _build_orchestrator() -> ContractOrchestrator:
    classifier = HFClassifier(model_name_or_path=CLASSIFIER_MODEL)
    extractor = _build_extractor()
    vector_db = ChromaWrapper(persist_directory=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
    llm_provider = _build_llm_provider()
    policy = RiskPolicy()  # ships with full 41-category default policy

    return ContractOrchestrator(
        classifier=classifier,
        risk_policy=policy,
        extractor=extractor,
        llm_provider=llm_provider,
        vector_db=vector_db,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifespan: build orchestrator + job infrastructure at startup."""
    global orchestrator, report_generator, job_store, job_executor
    configure_logging()
    warn_if_insecure()

    # Job store is built even in TESTING mode so the upload tests can use it
    # with a mocked orchestrator. Path is overridden in tests via env.
    job_store = JobStore(db_path=default_db_path())
    job_executor = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="job-")
    logger.info("Job store ready at %s, %d workers.", default_db_path(), JOB_WORKERS)

    if _is_truthy(os.getenv("TESTING")):
        logger.info("TESTING mode — skipping orchestrator initialization.")
        yield
        job_executor.shutdown(wait=True)
        job_executor = None
        job_store = None
        return

    try:
        orchestrator = _build_orchestrator()
        report_generator = GenerateComplianceReport(output_dir=REPORTS_DIR)
        logger.info("Orchestrator initialized.")
    except Exception as exc:
        logger.exception(f"Failed to initialize orchestrator: {exc}")
        orchestrator = None
    yield
    orchestrator = None
    report_generator = None
    if job_executor:
        job_executor.shutdown(wait=True)
        job_executor = None
    job_store = None


app = FastAPI(
    title="ContractLens API",
    description="API for contract risk analysis using Multi-Agent RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter — must be attached before the SlowAPI middleware below.
# Endpoint-specific limits use @limiter.limit("..."); the limiter also
# applies any RATE_LIMIT_DEFAULTS env value as a global ceiling.
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return 429 JSON instead of slowapi's default plain-text 429."""
    from starlette.responses import JSONResponse

    detail = f"Rate limit exceeded: {exc.detail}"
    return JSONResponse({"detail": detail}, status_code=429)


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Middleware order matters: outermost runs first on the request and last
# on the response. Request ID -> metrics -> body size -> CORS -> SlowAPI -> route.
# Metrics is placed near the outside so it captures latency including the
# SlowAPI middleware's own work and the body-size rejection latency.
app.add_middleware(SlowAPIMiddleware)
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
    )
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    source_doc: Optional[str] = None


class RiskScoreResponse(BaseModel):
    category: str
    risk_level: str
    score: float
    justification: str
    extracted_span: str
    metadata: Dict[str, Any]
    span_start_offset: Optional[int] = None
    span_end_offset: Optional[int] = None
    source_doc: Optional[str] = None


class ReportRequest(BaseModel):
    text: str
    source_doc: Optional[str] = None
    format: str = "json"  # "json" or "pdf"


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str
    status_url: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    source_filename: Optional[str] = None
    source_format: Optional[str] = None
    char_count: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[List[RiskScoreResponse]] = None


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------
def get_orchestrator() -> ContractOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def get_report_generator() -> GenerateComplianceReport:
    if report_generator is None:
        raise HTTPException(status_code=503, detail="Report generator not initialized")
    return report_generator


def get_job_store() -> JobStore:
    if job_store is None:
        raise HTTPException(status_code=503, detail="Job store not initialized")
    return job_store


def get_job_executor() -> ThreadPoolExecutor:
    if job_executor is None:
        raise HTTPException(status_code=503, detail="Job executor not initialized")
    return job_executor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/analyze", response_model=List[RiskScoreResponse])
@limiter.limit(os.getenv("RATE_LIMIT_ANALYZE", "60/minute"))
async def analyze_contract(
    request: Request,
    response: Response,
    body: AnalyzeRequest,
    orch: ContractOrchestrator = Depends(get_orchestrator),
    _key: str = Depends(require_api_key),
):
    """Analyze a contract text block; return identified risk scores."""
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    risks = orch.analyze(body.text, source_doc=body.source_doc)

    return [
        RiskScoreResponse(
            category=r.category,
            risk_level=r.risk_level,
            score=r.score,
            justification=r.justification,
            extracted_span=r.extracted_span,
            metadata=r.metadata,
            span_start_offset=r.span_start_offset,
            span_end_offset=r.span_end_offset,
            source_doc=r.source_doc,
        )
        for r in risks
    ]


@app.post("/api/v1/report")
@limiter.limit(os.getenv("RATE_LIMIT_REPORT", "10/minute"))
async def generate_report(
    request: Request,
    response: Response,
    body: ReportRequest,
    orch: ContractOrchestrator = Depends(get_orchestrator),
    reporter: GenerateComplianceReport = Depends(get_report_generator),
    _key: str = Depends(require_api_key),
):
    """Analyze a contract text block and emit a compliance report (JSON or PDF)."""
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if body.format not in {"json", "pdf"}:
        raise HTTPException(status_code=400, detail="format must be 'json' or 'pdf'")

    risks = orch.analyze(body.text, source_doc=body.source_doc)
    contract = Contract(
        raw_text=body.text,
        metadata=ContractMetadata(
            source_path=body.source_doc or "inline",
            file_format="txt",
            char_count=len(body.text),
            analyzed_at=datetime.now(timezone.utc),
        ),
    )
    for r in risks:
        contract.add_risk(r)

    if body.format == "json":
        path = reporter.to_json(contract)
    else:
        path = reporter.to_pdf(contract)

    return {
        "report_path": str(path),
        "summary": contract.risk_summary(),
        "categories": contract.categories_present(),
        "n_risks": len(contract.risks),
    }


@app.post(
    "/api/v1/contracts",
    response_model=JobAcceptedResponse,
    status_code=202,
)
@limiter.limit(os.getenv("RATE_LIMIT_UPLOAD", "10/minute"))
async def upload_contract(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    orch: ContractOrchestrator = Depends(get_orchestrator),
    store: JobStore = Depends(get_job_store),
    executor: ThreadPoolExecutor = Depends(get_job_executor),
    api_key: str = Depends(require_api_key),
):
    """Accept a PDF/DOCX/TXT/MD upload; analyse in a background job.

    Returns 202 with a job_id immediately; poll GET /api/v1/jobs/{id} to
    retrieve status and (when completed) the RiskScore list.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename.")

    try:
        fmt = detect_format(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(file.filename).suffix.lower()
    job_id = store.create(
        api_key=api_key,
        source_filename=file.filename,
        source_format=fmt.value,
        char_count=len(file_bytes),
    )

    executor.submit(
        run_analysis_job,
        job_id=job_id,
        file_bytes=file_bytes,
        filename=file.filename,
        suffix=suffix,
        orchestrator=orch,
        job_store=store,
    )

    return JobAcceptedResponse(
        job_id=job_id,
        status=JobStatus.PENDING.value,
        status_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    request: Request,
    store: JobStore = Depends(get_job_store),
    api_key: str = Depends(require_api_key),
):
    """Return the current status (and result if completed) of a background job.

    Authorization: a job is only visible to the API key that submitted it.
    The 'auth-disabled' sentinel key (dev mode) can read every job.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    submitter = job.get("api_key")
    if submitter and api_key != "auth-disabled" and submitter != api_key:
        # Behave as 404 (not 403) to avoid leaking job existence to other tenants.
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return JobStatusResponse(
        id=job["id"],
        status=job["status"],
        source_filename=job.get("source_filename"),
        source_format=job.get("source_format"),
        char_count=job.get("char_count"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        result=[RiskScoreResponse(**r) for r in (job.get("result") or [])] or None,
    )


@app.get("/health")
async def health_check():
    """System health check: surfaces orchestrator readiness."""
    return {
        "status": (
            "ok" if orchestrator is not None or _is_truthy(os.getenv("TESTING")) else "degraded"
        ),
        "version": "1.0.0",
        "orchestrator_ready": orchestrator is not None,
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus exposition. No auth — Prometheus scrapers do not carry keys.

    Disable with METRICS_ENABLED=0 if the deployment exposes :8000 to the
    public internet without a network policy in front; otherwise the
    metrics are safe to expose (no secrets, just counters / histograms).
    """
    if not metrics_enabled():
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled.")
    return render_metrics()
