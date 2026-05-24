"""Tests for OpenAIProvider — fully mocked, no network."""

from unittest.mock import MagicMock, patch

import pytest

from src.application.interfaces.illm_provider import LLMMessage
from src.infrastructure.llm.openai_provider import OpenAIProvider


@pytest.fixture
def mock_openai():
    with (
        patch("src.infrastructure.llm.openai_provider.OpenAI") if False else patch("openai.OpenAI")
    ) as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


def _completion(content: str, prompt_tokens: int = 12, completion_tokens: int = 34):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock(message=msg)
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    comp = MagicMock(choices=[choice], usage=usage)
    return comp


def test_provider_returns_response_content_and_usage(mock_openai):
    mock_openai.chat.completions.create.return_value = _completion("Hello world")
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")

    resp = provider.chat(
        [LLMMessage(role="user", content="Say hi")],
        temperature=0.0,
        max_tokens=50,
    )

    assert resp.content == "Hello world"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage["prompt_tokens"] == 12
    assert resp.usage["completion_tokens"] == 34
    assert resp.usage["total_tokens"] == 46
    mock_openai.chat.completions.create.assert_called_once()


def test_provider_forwards_response_format_json_object(mock_openai):
    mock_openai.chat.completions.create.return_value = _completion('{"k": 1}')
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")

    provider.chat(
        [LLMMessage(role="user", content="json please")],
        response_format="json_object",
    )

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_provider_rejects_empty_message_list(mock_openai):
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    with pytest.raises(ValueError, match="at least one"):
        provider.chat([])


def test_provider_name_includes_model(mock_openai):
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")
    assert provider.name() == "openai:gpt-4o"
