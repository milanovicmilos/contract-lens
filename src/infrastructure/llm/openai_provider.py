"""
OpenAI implementation of ILLMProvider.

Uses the official `openai` Python SDK (v1+) directly rather than going
through langchain, which keeps this module a thin adapter and avoids
silent behavioural changes when langchain bumps versions. The default
model is `gpt-4o-mini` because it sits at the best price/quality point
for our two LLM workloads: per-clause legal commentary in the Consultant
agent and RAGAS faithfulness scoring in the evaluator.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.application.interfaces.illm_provider import ILLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class OpenAIProvider(ILLMProvider):
    """Concrete ILLMProvider backed by the OpenAI chat completions API.

    Construction is explicit about the API key so callers can pass a key from
    a secrets manager rather than relying on the OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed; run `pip install openai>=1.0`."
            ) from exc

        self._model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
            max_retries=max_retries,
        )

    def name(self) -> str:
        return f"openai:{self._model}"

    def chat(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must contain at least one entry")

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**payload)
        except Exception as exc:
            logger.error(f"OpenAI chat completion failed for model={self._model}: {exc}")
            raise

        choice = completion.choices[0]
        content = choice.message.content or ""
        usage_obj = getattr(completion, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
        }
        return LLMResponse(content=content, model=self._model, usage=usage)
