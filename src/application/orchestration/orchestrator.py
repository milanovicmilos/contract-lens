"""
LangGraph Orchestrator for ContractLens.

State machine that coordinates the four agents defined in the spec:
  Extractor -> Validator -> Legal Consultant -> Risk Auditor

Design notes:
- The LLM provider is optional. When absent, the Legal Consultant step
  degrades gracefully to a rule-based justification instead of raising, so the
  system remains usable in local-only / offline mode.
- The Vector DB is optional. When provided, the Consultant retrieves the top-k
  most-relevant regulatory snippets via RAG before generating its analysis.
- The Extractor is invoked per detected category to obtain exact span offsets,
  which feed into RiskScore traceability fields.
- The Validator combines two heuristics: minimum-length gate and a regex
  header-detector (matches all-caps section titles, "ARTICLE", "SECTION", etc.).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.application.interfaces.iclassifier import IClassifier
from src.application.interfaces.iextractor import ExtractionResult, IExtractor
from src.application.interfaces.illm_provider import ILLMProvider
from src.application.interfaces.ivector_db import IVectorDatabase
from src.application.llm_justifier import build_justifications
from src.domain.risk_policy import RiskPolicy
from src.domain.risk_score import RiskScore

logger = logging.getLogger(__name__)


# Regex heuristics for header / non-clause detection.
# Headers are typically short, all-caps, or start with structural markers.
HEADER_PATTERNS = [
    re.compile(r"^\s*(?:ARTICLE|SECTION|EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\b", re.IGNORECASE),
    re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*$"),  # bare numbering like "1.2.3"
    re.compile(r"^\s*[A-Z][A-Z\s]{4,}$"),  # all-caps headers
]

# Global classifier confidence floor. Used when the classifier does not ship a
# per-category thresholds.json (e.g., un-tuned HF Hub models). When the
# classifier *does* expose ``per_category_thresholds`` the per-category value
# is used instead — matching the conditions under which micro F1 = 0.688 was
# measured in RESULTS.md §1.
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.55"))
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
    # Per-category, RAG-grounded justifications from the LLM. Empty dict when
    # llm_provider is None or the call failed; the auditor then falls back to
    # the rule-based RiskPolicy justification per category.
    llm_justifications: Dict[str, str]
    final_risks: List[RiskScore]
    metadata: Dict[str, Any]


class ContractOrchestrator:
    """
    Coordinates the multi-agent pipeline using LangGraph.

    Required dependencies: classifier, risk_policy.
    Optional dependencies: extractor (for span localization), llm_provider
    (for richer consultant analysis), vector_db (for RAG retrieval).
    """

    def __init__(
        self,
        classifier: IClassifier,
        risk_policy: RiskPolicy,
        extractor: Optional[IExtractor] = None,
        llm_provider: Optional[ILLMProvider] = None,
        vector_db: Optional[IVectorDatabase] = None,
    ) -> None:
        self.classifier = classifier
        self.extractor = extractor
        self.risk_policy = risk_policy
        self.vector_db = vector_db
        self.llm_provider: Optional[ILLMProvider] = llm_provider

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
    def _threshold_for(self, category: str) -> float:
        """Return the per-category threshold when the classifier exposes one, else the global."""
        per_cat = getattr(self.classifier, "per_category_thresholds", None)
        if per_cat and category in per_cat:
            return per_cat[category]
        return CLASSIFICATION_THRESHOLD

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
        positive_cats = {
            c: s for c, s in classifications.items() if s >= self._threshold_for(c)
        }

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
        """Retrieve RAG context, then build per-category grounded justifications via LLM.

        Replaces the previous "single analysis blob appended to every risk"
        approach with a structured per-category dict from
        ``llm_justifier.build_justifications``. The change directly targets
        the RAGAS faithfulness ceiling documented in
        ``docs/RESULTS.md §3``.

        Graceful degradation: when ``llm_provider`` is None, the LLM call
        fails, or the JSON response is malformed, ``llm_justifications`` is
        an empty dict and the auditor falls back to rule-based
        justifications. We still emit the legacy ``consultant_analysis``
        marker string so downstream callers / tests that inspect that field
        keep working.
        """
        text = state.get("original_text", "")
        classifications = state.get("classifications", {}) or {}
        positive_cats = {
            c: s for c, s in classifications.items() if s >= self._threshold_for(c)
        }

        rag_context: List[Dict[str, Any]] = []
        if self.vector_db is not None:
            try:
                rag_context = self.vector_db.search(text, top_k=5)
            except Exception as exc:
                logger.warning(f"RAG retrieval failed: {exc}")

        if self.llm_provider is None:
            return {
                "consultant_analysis": (
                    "Local rule-based analysis only (LLM provider not configured)."
                ),
                "rag_context": rag_context,
                "llm_justifications": {},
            }

        # Build the per-category map (category -> risk_level) so the LLM can
        # tailor each justification. Falls back to "Medium" when the policy
        # rule isn't registered for a category.
        categories_with_levels: Dict[str, str] = {}
        for cat in positive_cats:
            level, _score, _just = self.risk_policy.assess_risk(cat, text)
            categories_with_levels[cat] = level

        llm_justifications = build_justifications(
            llm_provider=self.llm_provider,
            chunk_text=text,
            categories_with_levels=categories_with_levels,
            rag_context=rag_context,
        )

        return {
            "consultant_analysis": (
                f"LLM-grounded justifications produced for "
                f"{len(llm_justifications)}/{len(positive_cats)} categories."
            ),
            "rag_context": rag_context,
            "llm_justifications": llm_justifications,
        }

    def _node_audit(self, state: AgentState) -> dict:
        """Combine classifier, policy, extractor, and consultant outputs into RiskScores.

        Per-category justification source-of-truth:
          1. If ``llm_justifications[category]`` exists, use that — it was
             produced with the verbatim clause + RAG context and is what
             lifts RAGAS faithfulness past the rule-based ceiling.
          2. Otherwise, fall back to ``RiskPolicy.assess_risk`` (rule-based
             template with verbatim keyword quote). The fallback is what
             ships when llm_provider is None or the LLM call failed.

        Risk level and score still come from the RiskPolicy — the LLM does
        not get to invent risk levels. Only the explanation text is
        substituted.
        """
        text = state.get("original_text", "")
        source_doc = state.get("source_doc")
        classifications = state.get("classifications", {}) or {}
        spans = state.get("extracted_spans", {}) or {}
        llm_justifications: Dict[str, str] = state.get("llm_justifications", {}) or {}
        rag_hits = len(state.get("rag_context", []) or [])

        final_risks: List[RiskScore] = []
        for category, conf in classifications.items():
            if conf < self._threshold_for(category):
                continue

            level, policy_score, rule_justification = self.risk_policy.assess_risk(category, text)

            # Prefer the LLM-grounded justification when present; otherwise
            # fall back to the rule-based template. Metadata records which
            # path produced the text so downstream analytics can attribute.
            llm_text = llm_justifications.get(category, "").strip()
            if llm_text:
                justification = llm_text
                justification_source = "llm"
            else:
                justification = rule_justification
                justification_source = "rule"

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

            final_risks.append(
                RiskScore(
                    category=category,
                    risk_level=level,
                    score=policy_score,
                    justification=justification,
                    extracted_span=extracted_span_text,
                    metadata={
                        "classifier_confidence": conf,
                        "rag_hits": rag_hits,
                        "justification_source": justification_source,
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
            "llm_justifications": {},
            "is_valid_clause": False,
            "validation_reason": "",
            "consultant_analysis": "",
            "rag_context": [],
            "final_risks": [],
            "metadata": {},
        }
        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_risks", [])
