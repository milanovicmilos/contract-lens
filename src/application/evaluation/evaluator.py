"""
LLM-as-Judge evaluator implementing the RAGAS faithfulness pattern.

Faithfulness measures how well a generated justification is supported by the
original clause text. The classical RAGAS definition: extract atomic factual
claims from the answer, then check whether each claim can be verified from
the source. We approximate that with a single structured LLM prompt that
returns a 0..1 score plus an explicit reasoning trace.

The evaluator depends only on the ILLMProvider Strategy port, so tests can
inject a fake provider returning a canned JSON payload without pulling in
any vendor SDK.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict

from src.application.interfaces.illm_provider import ILLMProvider, LLMMessage
from src.domain.risk_score import RiskScore

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


class LLMEvaluator:
    """Evaluates RAG and Consultant outputs using LLM-as-a-judge patterns.

    The evaluator depends on the vendor-neutral ILLMProvider Strategy port,
    so tests can substitute a fake provider that returns a canned JSON
    response without any vendor SDK in the loop.
    """

    def __init__(self, llm_provider: ILLMProvider):
        if llm_provider is None:
            raise ValueError("llm_provider is required for LLM-as-a-judge evaluation.")
        self.llm_provider = llm_provider

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm_provider.chat(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            temperature=0.0,
            max_tokens=200,
            response_format="json_object",
        )
        return response.content

    def evaluate_faithfulness(self, risk_score: RiskScore, original_text: str) -> EvaluationResult:
        """Score how well risk_score.justification is supported by original_text."""
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

    def evaluate_relevancy(self, category: str, extracted_span: str) -> EvaluationResult:
        """Score whether extracted_span actually relates to category."""
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
