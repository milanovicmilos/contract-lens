"""
FastAPI entry point for ContractLens.

Startup is lenient: missing OPENAI_API_KEY only disables the Legal Consultant
LLM call (the rest of the pipeline still works in local-only mode). Model
selection and provider wiring is configurable via env vars so production can
override the dev defaults (small models for cold-start speed in CI / local dev).
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.middleware import BodySizeLimitMiddleware, RequestIDMiddleware
from src.api.rate_limit import limiter
from src.api.security import require_api_key, warn_if_insecure
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
    """FastAPI lifespan: build orchestrator at startup unless TESTING=True."""
    global orchestrator, report_generator
    warn_if_insecure()
    if _is_truthy(os.getenv("TESTING")):
        logger.info("TESTING mode — skipping orchestrator initialization.")
        yield
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
# on the response. Request ID -> body size -> CORS -> SlowAPI -> route.
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
