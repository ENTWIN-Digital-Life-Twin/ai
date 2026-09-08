from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    provider: str


class LLMProvider(Protocol):
    """Hides the concrete LLM backend (Ollama today, another HTTP API later)."""

    provider_name: str
    model_name: str

    def generate(self, messages: list[ChatMessage]) -> LLMResult:
        """Return assistant text. Must not persist prompts or user content."""
