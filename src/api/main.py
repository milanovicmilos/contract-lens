"""
FastAPI entry point for ContractLens.

Startup is lenient: missing OPENAI_API_KEY only disables the Legal Consultant
LLM call (the rest of the pipeline still works in local-only mode). Model
selection is configurable via env vars so production can override the dev
defaults (small models for cold-start speed in CI / local dev).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.domain.risk_policy import RiskPolicy
from src.infrastructure.agents.orchestrator import ContractOrchestrator
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DISABLE_LLM = _is_truthy(os.getenv("DISABLE_LLM"))


orchestrator: Optional[ContractOrchestrator] = None


def _build_llm_client():
    """Try to construct a langchain ChatOpenAI client; return None on any failure."""
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
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, openai_api_key=api_key, temperature=0.0)
    except Exception as exc:
        logger.error(f"Failed to initialize ChatOpenAI: {exc}; running without LLM.")
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
    llm_client = _build_llm_client()
    policy = RiskPolicy()  # ships with full 41-category default policy

    return ContractOrchestrator(
        classifier=classifier,
        risk_policy=policy,
        extractor=extractor,
        llm_client=llm_client,
        vector_db=vector_db,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI lifespan: build orchestrator at startup unless TESTING=True."""
    global orchestrator
    if _is_truthy(os.getenv("TESTING")):
        logger.info("TESTING mode — skipping orchestrator initialization.")
        yield
        return

    try:
        orchestrator = _build_orchestrator()
        logger.info("Orchestrator initialized.")
    except Exception as exc:
        logger.exception(f"Failed to initialize orchestrator: {exc}")
        orchestrator = None
    yield
    orchestrator = None


app = FastAPI(
    title="ContractLens API",
    description="API for contract risk analysis using Multi-Agent RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


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


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------
def get_orchestrator() -> ContractOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/analyze", response_model=List[RiskScoreResponse])
async def analyze_contract(
    request: AnalyzeRequest,
    orch: ContractOrchestrator = Depends(get_orchestrator),
):
    """Analyze a contract text block; return identified risk scores."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    risks = orch.analyze(request.text, source_doc=request.source_doc)

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


@app.get("/health")
async def health_check():
    """System health check: surfaces orchestrator readiness."""
    return {
        "status": "ok" if orchestrator is not None or _is_truthy(os.getenv("TESTING")) else "degraded",
        "version": "1.0.0",
        "orchestrator_ready": orchestrator is not None,
    }
