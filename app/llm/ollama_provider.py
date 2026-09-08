from __future__ import annotations

import logging

import httpx

from app.core.exceptions import LLMUnavailableError
from app.llm.provider import ChatMessage, LLMResult

logger = logging.getLogger("entwin.ai.llm")


class OllamaProvider:
    """Remote Ollama HTTP client. Does not start or bundle a local Ollama process."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        connect_timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[ChatMessage]) -> LLMResult:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError("Ollama server is unreachable.") from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Ollama request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("Ollama request failed.") from exc

        if response.status_code >= 400:
            logger.warning(
                "ollama_http_error status=%s model=%s",
                response.status_code,
                self._model,
            )
            raise LLMUnavailableError("Ollama rejected the chat request.")

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMUnavailableError("Ollama returned an invalid response.") from exc

        message = body.get("message") if isinstance(body, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError("Ollama returned an empty assistant message.")

        model = str(body.get("model") or self._model)
        logger.info("llm_success provider=ollama model=%s", model)
        return LLMResult(text=text.strip(), model=model, provider=self.provider_name)
