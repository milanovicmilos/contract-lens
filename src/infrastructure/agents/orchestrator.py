"""
LangGraph Orchestrator for ContractLens.

State machine that coordinates the four agents defined in the spec:
  Extractor -> Validator -> Legal Consultant -> Risk Auditor

Design notes:
- The LLM client is optional. When absent, the Legal Consultant step degrades
  gracefully to a rule-based justification instead of raising, so the system
  remains usable in local-only / offline mode.
- The Vector DB is optional. When provided, the Consultant retrieves the top-k
  most-relevant regulatory snippets via RAG before generating its analysis.
- The Extractor is invoked per detected category to obtain exact span offsets,
  which feed into RiskScore traceability fields.
- The Validator combines two heuristics: minimum-length gate and a regex
  header-detector (matches all-caps section titles, "ARTICLE", "SECTION", etc.).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.application.interfaces.iclassifier import IClassifier
from src.application.interfaces.iextractor import ExtractionResult, IExtractor
from src.application.interfaces.irisk_analyzer import RiskScore
from src.application.interfaces.ivector_db import IVectorDatabase
from src.domain.risk_policy import RiskPolicy

logger = logging.getLogger(__name__)


# Regex heuristics for header / non-clause detection.
# Headers are typically short, all-caps, or start with structural markers.
HEADER_PATTERNS = [
    re.compile(r"^\s*(?:ARTICLE|SECTION|EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\b", re.IGNORECASE),
    re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*$"),  # bare numbering like "1.2.3"
    re.compile(r"^\s*[A-Z][A-Z\s]{4,}$"),  # all-caps headers
]

CLASSIFICATION_THRESHOLD = 0.5
MIN_CLAUSE_WORDS = 8


class AgentState(TypedDict, total=False):
    """Internal state passed between graph nodes."""

    original_text: str
    source_doc: Optional[str]
    classifications: Dict[str, float]
    extracted_spans: Dict[str, List[ExtractionResult]]
    is_valid_clause: bool
    validation_reason: str
    consultant_analysis: str
    rag_context: List[Dict[str, Any]]
    final_risks: List[RiskScore]
    metadata: Dict[str, Any]


class ContractOrchestrator:
    """
    Coordinates the multi-agent pipeline using LangGraph.

    Required dependencies: classifier, risk_policy.
    Optional dependencies: extractor (for span localization), llm_client
    (for richer consultant analysis), vector_db (for RAG retrieval).
    """

    def __init__(
        self,
        classifier: IClassifier,
        risk_policy: RiskPolicy,
        extractor: Optional[IExtractor] = None,
        llm_client: Any = None,
        vector_db: Optional[IVectorDatabase] = None,
    ) -> None:
        self.classifier = classifier
        self.extractor = extractor
        self.risk_policy = risk_policy
        self.llm_client = llm_client
        self.vector_db = vector_db
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("extractor_agent", self._node_extract)
        workflow.add_node("validator_agent", self._node_validate)
        workflow.add_node("consultant_agent", self._node_consult)
        workflow.add_node("auditor_agent", self._node_audit)

        workflow.add_edge(START, "extractor_agent")
        workflow.add_edge("extractor_agent", "validator_agent")
        workflow.add_conditional_edges(
            "validator_agent",
            self._route_after_validation,
            {"continue": "consultant_agent", "end": END},
        )
        workflow.add_edge("consultant_agent", "auditor_agent")
        workflow.add_edge("auditor_agent", END)

        return workflow.compile()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _is_header(text: str) -> bool:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        for pattern in HEADER_PATTERNS:
            if pattern.match(first_line):
                return True
        return False

    # ----------------------------------------------------------------- nodes
    def _node_extract(self, state: AgentState) -> dict:
        """Classify the text and (if extractor available) localize each positive category."""
        text = state.get("original_text", "")
        if not text.strip():
            return {"classifications": {}, "extracted_spans": {}}

        classifications = self.classifier.classify(text)
        positive_cats = {c: s for c, s in classifications.items() if s >= CLASSIFICATION_THRESHOLD}

        spans: Dict[str, List[ExtractionResult]] = {}
        if self.extractor and positive_cats:
            for category in positive_cats:
                try:
                    question = (
                        f'Highlight the parts (if any) of this contract related to "{category}".'
                    )
                    results = self.extractor.extract(context=text, question=question, top_k=1)
                    spans[category] = results
                except Exception as exc:
                    logger.warning(f"Extractor failed for category '{category}': {exc}")
                    spans[category] = []

        logger.debug(
            f"Extract: {len(positive_cats)} positive categories, "
            f"{sum(len(v) for v in spans.values())} spans"
        )
        return {"classifications": classifications, "extracted_spans": spans}

    def _node_validate(self, state: AgentState) -> dict:
        """Validate whether the input looks like a real clause vs. a header / noise."""
        text = state.get("original_text", "").strip()

        if not text:
            return {"is_valid_clause": False, "validation_reason": "empty text"}

        if len(text.split()) < MIN_CLAUSE_WORDS:
            return {
                "is_valid_clause": False,
                "validation_reason": f"too short (<{MIN_CLAUSE_WORDS} words)",
            }

        if self._is_header(text):
            return {"is_valid_clause": False, "validation_reason": "looks like a header"}

        return {"is_valid_clause": True, "validation_reason": "passes heuristic checks"}

    def _route_after_validation(self, state: AgentState) -> str:
        return "continue" if state.get("is_valid_clause", False) else "end"

    def _node_consult(self, state: AgentState) -> dict:
        """Run RAG retrieval + LLM consultation, or fall back to rule-based note."""
        text = state.get("original_text", "")

        rag_context: List[Dict[str, Any]] = []
        if self.vector_db is not None:
            try:
                rag_context = self.vector_db.search(text, top_k=3)
            except Exception as exc:
                logger.warning(f"RAG retrieval failed: {exc}")

        if self.llm_client is None:
            note = (
                "Local rule-based analysis only (LLM client not configured). "
                "Review against policy keywords for risk indicators."
            )
            return {"consultant_analysis": note, "rag_context": rag_context}

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            context_block = ""
            if rag_context:
                snippets = "\n\n".join(
                    f"[{i+1}] {r.get('text', '')[:500]}" for i, r in enumerate(rag_context)
                )
                context_block = f"\n\nRELEVANT REGULATIONS:\n{snippets}"

            prompt = [
                SystemMessage(
                    content=(
                        "You are an expert legal consultant. Provide a concise "
                        "(1-2 sentences) analysis of the risks present in the following "
                        "contract clause. Cite specific regulations from RELEVANT REGULATIONS "
                        "when applicable. If the clause is standard and non-hazardous, say so."
                    )
                ),
                HumanMessage(content=f"Clause text:\n\n{text}{context_block}"),
            ]
            response = self.llm_client.invoke(prompt)
            analysis = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning(f"LLM consultation failed: {exc}; falling back to rule-based note.")
            analysis = (
                "LLM analysis unavailable due to runtime error. "
                "Risk score reflects policy-based assessment only."
            )

        return {"consultant_analysis": analysis, "rag_context": rag_context}

    def _node_audit(self, state: AgentState) -> dict:
        """Combine classifier, policy, extractor, and consultant outputs into RiskScores."""
        text = state.get("original_text", "")
        source_doc = state.get("source_doc")
        classifications = state.get("classifications", {}) or {}
        spans = state.get("extracted_spans", {}) or {}
        analysis = state.get("consultant_analysis", "") or ""

        final_risks: List[RiskScore] = []
        for category, conf in classifications.items():
            if conf < CLASSIFICATION_THRESHOLD:
                continue

            level, policy_score, justification = self.risk_policy.assess_risk(category, text)

            category_spans = spans.get(category, [])
            if category_spans:
                top = category_spans[0]
                extracted_span_text = top.text or text
                span_start = top.answer_start if top.answer_start >= 0 else None
                span_end = top.answer_end if top.answer_end >= 0 else None
            else:
                extracted_span_text = text
                span_start = None
                span_end = None

            enhanced_justification = (
                f"{justification} Consultant Note: {analysis}" if analysis else justification
            )

            final_risks.append(
                RiskScore(
                    category=category,
                    risk_level=level,
                    score=policy_score,
                    justification=enhanced_justification,
                    extracted_span=extracted_span_text,
                    metadata={
                        "classifier_confidence": conf,
                        "rag_hits": len(state.get("rag_context", []) or []),
                    },
                    span_start_offset=span_start,
                    span_end_offset=span_end,
                    source_doc=source_doc,
                )
            )

        return {"final_risks": final_risks}

    # ----------------------------------------------------------------- entry
    def analyze(self, text: str, source_doc: Optional[str] = None) -> List[RiskScore]:
        """Run the full pipeline on a text block; return the produced RiskScores."""
        initial_state: AgentState = {
            "original_text": text,
            "source_doc": source_doc,
            "classifications": {},
            "extracted_spans": {},
            "is_valid_clause": False,
            "validation_reason": "",
            "consultant_analysis": "",
            "rag_context": [],
            "final_risks": [],
            "metadata": {},
        }
        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_risks", [])
