"""
Tests for src/application/llm_justifier.py — LLM-grounded justifications.

Strategy: build a fake ``ILLMProvider`` that returns whatever JSON the
test wants, and assert ``build_justifications`` correctly parses,
filters, and degrades.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.application.interfaces.illm_provider import LLMResponse
from src.application.llm_justifier import (
    _build_user_prompt,
    _extract_json,
    build_justifications,
)


def _fake_provider(content: str):
    """Return a MagicMock standing in for ILLMProvider; .chat returns LLMResponse(content)."""
    prov = MagicMock()
    prov.chat.return_value = LLMResponse(content=content, model="fake")
    return prov


# ---------------------------------------------------------------- happy path
def test_build_justifications_returns_per_category_text():
    prov = _fake_provider(
        '{"justifications": '
        '{"Cap On Liability": "Per Practice Note on liability caps, the clause '
        '\\"capped at $1,000\\" sets the limit at a fixed sum.", '
        '"Termination For Convenience": "Per Practice Note on termination, '
        '\\"thirty days written notice\\" allows wind-down."}}'
    )
    result = build_justifications(
        llm_provider=prov,
        chunk_text=(
            "Either party may terminate for convenience upon thirty days written notice. "
            "Liability is capped at $1,000."
        ),
        categories_with_levels={
            "Cap On Liability": "Medium",
            "Termination For Convenience": "Medium",
        },
        rag_context=[
            {
                "text": "Liability caps should be reasonable...",
                "metadata": {"source": "Practice Note", "title": "Liability"},
            }
        ],
    )
    assert set(result.keys()) == {"Cap On Liability", "Termination For Convenience"}
    assert '"capped at $1,000"' in result["Cap On Liability"]
    assert "thirty days" in result["Termination For Convenience"]
    prov.chat.assert_called_once()


def test_build_justifications_filters_invented_categories():
    """If the LLM hallucinates a category we did not ask about, drop it silently —
    we never propagate an unrequested key onto a RiskScore."""
    prov = _fake_provider(
        '{"justifications": {'
        '"Cap On Liability": "real justification",'
        '"Made Up Category": "should be filtered out"'
        "}}"
    )
    result = build_justifications(
        llm_provider=prov,
        chunk_text="some clause",
        categories_with_levels={"Cap On Liability": "Medium"},
    )
    assert list(result.keys()) == ["Cap On Liability"]


# --------------------------------------------------------------- degradation
def test_no_provider_returns_empty_dict():
    """No LLM provider configured → empty dict (orchestrator falls back to rule-based)."""
    result = build_justifications(
        llm_provider=None,
        chunk_text="anything",
        categories_with_levels={"Cap On Liability": "Medium"},
    )
    assert result == {}


def test_empty_categories_returns_empty_dict():
    """No positive categories → no LLM call, no work."""
    prov = _fake_provider("{}")
    result = build_justifications(
        llm_provider=prov,
        chunk_text="anything",
        categories_with_levels={},
    )
    assert result == {}
    prov.chat.assert_not_called()


def test_llm_exception_falls_back_to_empty_dict():
    """Network / API error from the provider → orchestrator gets {} and uses rule-based."""
    prov = MagicMock()
    prov.chat.side_effect = RuntimeError("OpenAI 500")
    result = build_justifications(
        llm_provider=prov,
        chunk_text="clause",
        categories_with_levels={"Cap On Liability": "Medium"},
    )
    assert result == {}


def test_malformed_json_returns_empty_dict():
    """LLM returned prose instead of JSON → tolerated, fall back."""
    prov = _fake_provider("Sorry, I cannot answer that.")
    result = build_justifications(
        llm_provider=prov,
        chunk_text="clause",
        categories_with_levels={"Cap On Liability": "Medium"},
    )
    assert result == {}


def test_missing_justifications_key_returns_empty_dict():
    """LLM returned JSON without the expected 'justifications' key → tolerated."""
    prov = _fake_provider('{"wrong_key": {"Cap On Liability": "text"}}')
    result = build_justifications(
        llm_provider=prov,
        chunk_text="clause",
        categories_with_levels={"Cap On Liability": "Medium"},
    )
    assert result == {}


def test_non_string_justification_values_dropped():
    """If the LLM returns {category: 123} instead of {category: str}, drop that entry."""
    prov = _fake_provider(
        '{"justifications": {'
        '"Cap On Liability": "valid text",'
        '"Termination For Convenience": 12345'
        "}}"
    )
    result = build_justifications(
        llm_provider=prov,
        chunk_text="clause",
        categories_with_levels={
            "Cap On Liability": "Medium",
            "Termination For Convenience": "Medium",
        },
    )
    assert list(result.keys()) == ["Cap On Liability"]


# ----------------------------------------------------------------- internals
def test_user_prompt_includes_categories_and_levels():
    prompt = _build_user_prompt(
        chunk_text="the clause text",
        categories_with_levels={"Cap On Liability": "Medium", "Non-Compete": "High"},
        rag_context=None,
    )
    assert "the clause text" in prompt
    assert "Cap On Liability (risk level: Medium)" in prompt
    assert "Non-Compete (risk level: High)" in prompt
    assert "RELEVANT REGULATIONS" not in prompt  # no rag → no block


def test_user_prompt_includes_rag_identifiers_when_present():
    prompt = _build_user_prompt(
        chunk_text="clause",
        categories_with_levels={"Cap On Liability": "Medium"},
        rag_context=[
            {
                "text": "GDPR Art 28 says processors must do X.",
                "metadata": {"source": "GDPR", "article": "28"},
            }
        ],
    )
    assert "GDPR Art. 28" in prompt
    assert "RELEVANT REGULATIONS" in prompt


def test_user_prompt_caps_rag_at_five_snippets():
    """Don't blow up the prompt size on a giant RAG return."""
    rag = [
        {"text": f"snippet {i}", "metadata": {"source": "GDPR", "article": str(i)}}
        for i in range(10)
    ]
    prompt = _build_user_prompt(
        chunk_text="clause",
        categories_with_levels={"Cap On Liability": "Medium"},
        rag_context=rag,
    )
    # 5 snippets should appear; the 6th-10th should not.
    for i in range(5):
        assert f"Art. {i}" in prompt
    for i in range(5, 10):
        assert f"Art. {i}" not in prompt


def test_extract_json_tolerates_prose_surrounding():
    raw = 'Here\'s the answer:\n{"justifications": {"x": "y"}}\nThanks!'
    parsed = _extract_json(raw)
    assert parsed == {"justifications": {"x": "y"}}


def test_extract_json_returns_empty_on_invalid():
    assert _extract_json("") == {}
    assert _extract_json("no braces here") == {}
    assert _extract_json("{not json}") == {}


# ---------------------------------------------------- response_format requested
def test_chat_called_with_json_response_format():
    """We MUST pass response_format='json_object' so the LLM emits valid JSON."""
    prov = _fake_provider('{"justifications": {}}')
    build_justifications(
        llm_provider=prov,
        chunk_text="clause",
        categories_with_levels={"Cap On Liability": "Medium"},
    )
    call_kwargs = prov.chat.call_args.kwargs
    assert call_kwargs.get("response_format") == "json_object"
    assert call_kwargs.get("temperature") == 0.0
