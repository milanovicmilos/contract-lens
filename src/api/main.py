import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.domain.risk_policy import RiskPolicy
from src.infrastructure.agents.orchestrator import ContractOrchestrator
from src.infrastructure.ai.hf_classifier import HFClassifier
from src.infrastructure.database.chroma_wrapper import ChromaWrapper

load_dotenv()

app = FastAPI(
    title="ContractLens API",
    description="API for contract risk analysis using Multi-Agent RAG.",
    version="1.0.0",
)

# Application state
orchestrator: Optional[ContractOrchestrator] = None


@app.on_event("startup")
async def startup_event():
    global orchestrator
    if os.getenv("TESTING", "False") == "True":
        return

    # Initialize basic components
    classifier = HFClassifier(
        model_name_or_path="prajjwal1/bert-tiny"  # Koristimo mikro model (17MB) za instant lokalno testiranje umesto DeBERTa (874MB)
    )  # Will lazily load or use stub if no model downloaded
    extractor = None  # Optional for orchestration if not used directly
    ChromaWrapper(persist_directory="./chroma_db", collection_name="legal_regulations")

    # Real OpenAI Client from Langchain
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing. API cannot start.")

    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.0)

    policy = RiskPolicy(rules={})  # Default rules

    orchestrator = ContractOrchestrator(
        classifier=classifier, extractor=extractor, risk_policy=policy, llm_client=llm
    )


class AnalyzeRequest(BaseModel):
    text: str


class RiskScoreResponse(BaseModel):
    category: str
    risk_level: str
    score: float
    justification: str
    extracted_span: str
    metadata: Dict[str, Any]


def get_orchestrator() -> ContractOrchestrator:
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    return orchestrator


@app.post("/api/v1/analyze", response_model=List[RiskScoreResponse])
async def analyze_contract(
    request: AnalyzeRequest, orch: ContractOrchestrator = Depends(get_orchestrator)
):
    """
    Analyzes a contract text block and returns identified risk scores.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    risks = orch.analyze(request.text)

    # Map domain RiskScores to API RiskScoreResponses
    response = []
    for r in risks:
        response.append(
            RiskScoreResponse(
                category=r.category,
                risk_level=r.risk_level,
                score=r.score,
                justification=r.justification,
                extracted_span=r.extracted_span,
                metadata=r.metadata,
            )
        )
    return response


@app.get("/health")
async def health_check():
    """System health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}
