"""
LangGraph Orchestrator for ContractLens
Manages the state machine and coordinates agents: Extractor, Validator, LegalConsultant, RiskAuditor.
"""

import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.application.interfaces.iclassifier import IClassifier
from src.application.interfaces.iextractor import IExtractor
from src.application.interfaces.irisk_analyzer import RiskScore
from src.domain.risk_policy import RiskPolicy

logger = logging.getLogger(__name__)


# 1. Define the Graph State
class AgentState(TypedDict):
    """Represents the internal state of the agent workflow during processing."""

    original_text: str
    classifications: Dict[str, float]
    extracted_spans: List[str]
    is_valid_clause: bool
    consultant_analysis: str
    final_risks: List[RiskScore]
    metadata: Dict[str, Any]


class ContractOrchestrator:
    """
    Coordinates the execution of multiple agents/steps using LangGraph.
    """

    def __init__(
        self,
        classifier: IClassifier,
        extractor: IExtractor,
        risk_policy: RiskPolicy,
        llm_client: Any = None,  # Will be a langchain ChatOpenAI or similar
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.risk_policy = risk_policy
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("extractor_agent", self._node_extract)
        workflow.add_node("validator_agent", self._node_validate)
        workflow.add_node("consultant_agent", self._node_consult)
        workflow.add_node("auditor_agent", self._node_audit)

        # Add edges
        workflow.add_edge(START, "extractor_agent")
        workflow.add_edge("extractor_agent", "validator_agent")

        # Conditional edge from validator -> consultant OR end
        workflow.add_conditional_edges(
            "validator_agent",
            self._route_after_validation,
            {"continue": "consultant_agent", "end": END},
        )

        workflow.add_edge("consultant_agent", "auditor_agent")
        workflow.add_edge("auditor_agent", END)

        return workflow.compile()

    def _node_extract(self, state: AgentState) -> dict:
        """Node 1: Runs local extraction and classification."""
        logger.debug("Running Extractor Agent")

        # Dummy invocation using standard tools
        text = state.get("original_text", "")
        if not text.strip():
            return {"classifications": {}, "extracted_spans": []}

        # Use the classifier component from earlier phases
        classifications = self.classifier.classify(text)

        # Extractor would map spans here (for now we treat the block as the span)
        # spans = self.extractor.extract(...)
        spans = [text]

        return {"classifications": classifications, "extracted_spans": spans}

    def _node_validate(self, state: AgentState) -> dict:
        """Node 2: Validates if the text is a real clause or just noise/header."""
        logger.debug("Running Validator Agent")

        text = state.get("original_text", "")
        # Very naive heuristic for PoC: if it's too short, it's a header, not a clause
        is_valid = len(text.split()) > 5

        return {"is_valid_clause": is_valid}

    def _route_after_validation(self, state: AgentState) -> str:
        """Routing logic: If invalid, stop. If valid, consult LLM."""
        if state.get("is_valid_clause", False):
            return "continue"
        return "end"

    def _node_consult(self, state: AgentState) -> dict:
        """Node 3: Consults external/stub LLM for sophisticated reasoning (OpenAI/RAG)."""
        logger.debug("Running Legal Consultant Agent")

        # In a real setup, we would prompt self.llm_client and query ChromaDB
        analysis = "Legal consultant analyzed the terms and found standard operational parameters."

        return {"consultant_analysis": analysis}

    def _node_audit(self, state: AgentState) -> dict:
        """Node 4: Produces the final RiskScore using Domain policy."""
        logger.debug("Running Risk Auditor Agent")

        final_risks = []
        text = state.get("original_text", "")
        classifications = state.get("classifications", {})
        analysis = state.get("consultant_analysis", "")

        for category, score in classifications.items():
            if score > 0.5:  # threshold
                r_level, r_score, justify = self.risk_policy.assess_risk(category, text)

                # Enhance justification with consultant analysis
                enhanced_justify = f"{justify} Consultant Note: {analysis}"

                risk = RiskScore(
                    category=category,
                    risk_level=r_level,
                    score=r_score,
                    justification=enhanced_justify,
                    extracted_span=text,
                    metadata={"confidence": score},
                )
                final_risks.append(risk)

        return {"final_risks": final_risks}

    def analyze(self, text: str) -> List[RiskScore]:
        """Entry point to run the graph on a text block."""
        initial_state = AgentState(
            original_text=text,
            classifications={},
            extracted_spans=[],
            is_valid_clause=False,
            consultant_analysis="",
            final_risks=[],
            metadata={},
        )

        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_risks", [])
