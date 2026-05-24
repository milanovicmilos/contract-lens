"""
Strategy Pattern for LLM providers.

Decouples application/use-case code from any specific vendor SDK.
Today the only concrete implementation is OpenAI (src/infrastructure/llm/openai_provider.py),
but new providers (Anthropic, local llama.cpp, Azure OpenAI, ...) can be
added without touching the orchestrator or evaluator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLMMessage:
    """One turn of a chat-style prompt."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Provider-agnostic response container."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)


class ILLMProvider(ABC):
    """Provider-agnostic chat-completion interface."""

    @abstractmethod
    def chat(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,  # "text" or "json_object"
    ) -> LLMResponse:
        """Run a single chat completion. Implementations must be deterministic
        when temperature=0.0 and must not raise on benign content; only
        infrastructure failures (auth, network) should propagate.
        """

    @abstractmethod
    def name(self) -> str:
        """Stable identifier for the underlying model / provider."""
