"""
LLM-as-Judge evaluator implementing the RAGAS faithfulness pattern.

Faithfulness measures how well a generated justification is supported by the
original clause text. The classical RAGAS definition: extract atomic factual
claims from the answer, then check whether each claim can be verified from
the source. We approximate that with a single structured LLM prompt that
returns a 0..1 score plus an explicit reasoning trace.

This module remains testable without a real LLM by accepting any object that
exposes either:
- a callable `evaluate(original_text, justification) -> dict` (legacy / mock), or
- a langchain-style `invoke(messages) -> response` interface (real LLM client).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.application.interfaces.irisk_analyzer import RiskScore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    reasoning: str


FAITHFULNESS_SYSTEM_PROMPT = """You are a strict evaluator that judges whether \
a contract risk justification is faithful to the original clause text.

Rules:
1. A justification is FAITHFUL if every factual claim it makes can be verified \
from the original clause text.
2. A justification is UNFAITHFUL if it introduces facts not supported by the \
clause, contradicts the clause, or asserts unsubstantiated risks.
3. General commentary that doesn't make verifiable claims about the clause is \
considered NEUTRAL and scored 0.7 by default.

Output ONLY a JSON object with this exact schema:
{"faithfulness": <float between 0.0 and 1.0>, "reasoning": "<one-sentence justification>"}

No prose outside the JSON object."""

FAITHFULNESS_USER_TEMPLATE = """ORIGINAL CLAUSE:
{original_text}

JUSTIFICATION TO EVALUATE:
{justification}

Return your JSON verdict."""


RELEVANCY_SYSTEM_PROMPT = """You are a strict evaluator that judges whether \
an extracted text span is relevant to the stated category.

Score 1.0 if the span clearly demonstrates the category's concept.
Score 0.0 if the span is unrelated.
Score in between for partial relevance.

Output ONLY a JSON object: {"relevancy": <float>, "reasoning": "<one sentence>"}."""

RELEVANCY_USER_TEMPLATE = """CATEGORY: {category}

EXTRACTED SPAN:
{span}

Return your JSON verdict."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object out of an LLM response, tolerant of surrounding prose."""
    if not text:
        return {}
    match = _JSON_RE.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _has_method(obj: Any, name: str) -> bool:
    """True iff obj defines a real (non-auto-generated) attribute by that name.

    MagicMock auto-creates any attribute access, so plain hasattr() can't
    distinguish a deliberately-stubbed `.evaluate` from incidental access.
    We special-case MagicMock by consulting its child-mock registry.
    """
    if obj is None:
        return False
    try:
        from unittest.mock import MagicMock

        if isinstance(obj, MagicMock):
            return name in obj._mock_children or name in obj.__dict__
    except ImportError:
        pass
    return hasattr(obj, name)


class LLMEvaluator:
    """Evaluates RAG and Consultant outputs using LLM-as-a-judge patterns.

    Accepts either:
    - a legacy mock exposing `.evaluate(original_text, justification) -> dict`
    - a langchain-style chat client exposing `.invoke(messages) -> response`
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def _require_client(self) -> None:
        if self.llm_client is None:
            raise ValueError(
                "LLM client requires a valid instance for faithfulness evaluation."
            )

    def _is_legacy_evaluate_client(self) -> bool:
        return _has_method(self.llm_client, "evaluate")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch to the langchain `.invoke` interface for real LLM clients."""
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm_client.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return response.content if hasattr(response, "content") else str(response)

    def evaluate_faithfulness(
        self, risk_score: RiskScore, original_text: str
    ) -> EvaluationResult:
        """Score how well risk_score.justification is supported by original_text."""
        self._require_client()

        if self._is_legacy_evaluate_client():
            verdict = self.llm_client.evaluate(
                original_text=original_text, justification=risk_score.justification
            )
            return EvaluationResult(
                metric_name="faithfulness",
                score=float(verdict.get("faithfulness", 0.0)),
                reasoning=str(verdict.get("reasoning", "No reasoning provided.")),
            )

        user_prompt = FAITHFULNESS_USER_TEMPLATE.format(
            original_text=original_text.strip()[:4000],
            justification=risk_score.justification.strip()[:2000],
        )
        raw = self._call_llm(FAITHFULNESS_SYSTEM_PROMPT, user_prompt)
        if not isinstance(raw, str):
            raw = str(raw)
        parsed = _extract_json(raw)
        return EvaluationResult(
            metric_name="faithfulness",
            score=float(parsed.get("faithfulness", 0.0)),
            reasoning=str(parsed.get("reasoning", "Could not parse LLM response.")),
        )

    def evaluate_relevancy(
        self, category: str, extracted_span: str
    ) -> EvaluationResult:
        """Score whether extracted_span actually relates to category."""
        self._require_client()

        if self._is_legacy_evaluate_client():
            verdict = self.llm_client.evaluate(
                original_text=extracted_span, justification=category
            )
            return EvaluationResult(
                metric_name="relevancy",
                score=float(verdict.get("relevancy", verdict.get("faithfulness", 0.0))),
                reasoning=str(verdict.get("reasoning", "No reasoning provided.")),
            )

        user_prompt = RELEVANCY_USER_TEMPLATE.format(
            category=category, span=extracted_span.strip()[:2000]
        )
        raw = self._call_llm(RELEVANCY_SYSTEM_PROMPT, user_prompt)
        if not isinstance(raw, str):
            raw = str(raw)
        parsed = _extract_json(raw)
        return EvaluationResult(
            metric_name="relevancy",
            score=float(parsed.get("relevancy", 0.0)),
            reasoning=str(parsed.get("reasoning", "Could not parse LLM response.")),
        )
