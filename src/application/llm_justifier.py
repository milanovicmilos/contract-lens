"""
LLM-rewritten, RAG-grounded justifications for emitted RiskScores.

Replaces the rule-based ``RiskPolicy`` template (which caps RAGAS
faithfulness around 0.23 — see docs/RESULTS.md §3) with a single LLM
call per chunk that produces a structured ``Dict[category, justification]``.

Design choices captured here:

- **One LLM call per chunk, not per risk.** A 50-page contract with
  10 risks per chunk would cost 10x more LLM money for marginally
  better text otherwise. Batching the per-category justifications into
  one JSON-mode response keeps the cost flat.
- **JSON-mode response.** The LLM is asked to emit
  ``{"justifications": {"<category>": "<grounded text>"}}``. Parsing is
  tolerant: malformed JSON or missing keys fall back per-category to
  the rule-based template — we never propagate a hallucinated
  justification because parsing succeeded.
- **The prompt forces citation discipline.** The LLM is told (a) to
  quote a short verbatim span from the clause, (b) to cite the
  specific RAG article identifier when relevant, and (c) to NEVER
  introduce a fact not present in the clause or in the RAG context.
  Faithfulness checks downstream measure exactly this.
- **Graceful degradation.** If ``llm_provider`` is ``None`` or the
  call raises, return an empty dict — the orchestrator's auditor will
  use the rule-based justification for every category. The system
  still produces RiskScores; just at the lower faithfulness ceiling
  the rule-based path documents.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.application.interfaces.illm_provider import ILLMProvider, LLMMessage

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


SYSTEM_PROMPT = """You are a careful legal analyst. For each risk category detected in a contract \
clause, you write ONE concise justification (1-2 sentences max) that satisfies all of these rules:

1. The justification MUST quote a short verbatim phrase from the CLAUSE TEXT — copy the literal \
words, inside double-quotes, no paraphrasing.
2. When a RELEVANT REGULATION snippet is provided that supports the category, you MUST cite it \
by its identifier (e.g. "GDPR Art. 28", "EU AI Act Art. 5") at the start of your explanation.
3. You MUST NOT introduce any fact that is not present in CLAUSE TEXT or in RELEVANT REGULATIONS.
4. You MUST NOT speculate about the parties' intent, future actions, or jurisdiction-specific \
case law.
5. Output ONLY valid JSON with this exact schema:

   {"justifications": {"<category name>": "<your 1-2 sentence justification>"}}

   The category names you receive in the user prompt MUST appear verbatim as JSON keys. No prose \
outside the JSON object."""


def _build_user_prompt(
    chunk_text: str,
    categories_with_levels: Dict[str, str],
    rag_context: Optional[List[Dict[str, Any]]],
) -> str:
    """Compose the per-chunk user prompt — categories with their risk levels +
    optional RAG snippets the LLM should cite when relevant.
    """
    cat_lines = "\n".join(
        f"- {cat} (risk level: {level})" for cat, level in categories_with_levels.items()
    )

    rag_block = ""
    if rag_context:
        snippets: List[str] = []
        for r in rag_context[:5]:  # cap to 5 to keep prompt size sane
            meta = r.get("metadata") or {}
            source = meta.get("source") or ""
            article = meta.get("article") or ""
            ident = (
                f"{source} Art. {article}" if source and article else (source or "Practice Note")
            )
            text = (r.get("text") or "").strip()
            if not text:
                continue
            snippets.append(f"[{ident}] {text[:600]}")
        if snippets:
            rag_block = "\n\nRELEVANT REGULATIONS (cite by identifier when applicable):\n" + (
                "\n\n".join(snippets)
            )

    return (
        f"CLAUSE TEXT:\n{chunk_text.strip()[:4000]}\n\n"
        f"RISK CATEGORIES DETECTED IN THIS CLAUSE:\n{cat_lines}"
        f"{rag_block}\n\n"
        "Return the JSON object as specified."
    )


def _extract_json(raw: str) -> Dict[str, Any]:
    """Tolerant JSON parser — strips surrounding prose, returns {} on failure."""
    if not raw:
        return {}
    match = _JSON_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def build_justifications(
    *,
    llm_provider: Optional[ILLMProvider],
    chunk_text: str,
    categories_with_levels: Dict[str, str],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Dict[str, str]:
    """Return ``{category: grounded justification}`` for the chunk.

    Returns an empty dict on any failure (no LLM, network error, malformed
    JSON, missing 'justifications' key). The empty dict signals the
    orchestrator's auditor to use rule-based justifications.
    """
    if llm_provider is None or not categories_with_levels:
        return {}

    user_prompt = _build_user_prompt(chunk_text, categories_with_levels, rag_context)

    try:
        response = llm_provider.chat(
            messages=[
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
        )
    except Exception as exc:
        logger.warning("LLM justifier call failed: %s. Falling back to rule-based.", exc)
        return {}

    parsed = _extract_json(response.content)
    if not parsed:
        logger.warning(
            "LLM justifier returned unparseable content (%d chars). Falling back.",
            len(response.content or ""),
        )
        return {}

    justifications = parsed.get("justifications") or parsed
    if not isinstance(justifications, dict):
        logger.warning(
            "LLM justifier 'justifications' field is not a dict (%s). Falling back.",
            type(justifications).__name__,
        )
        return {}

    # Filter to only the categories we asked about — guards against the LLM
    # inventing new keys (which would silently end up on RiskScores).
    return {
        cat: str(text).strip()
        for cat, text in justifications.items()
        if cat in categories_with_levels and isinstance(text, str) and text.strip()
    }
