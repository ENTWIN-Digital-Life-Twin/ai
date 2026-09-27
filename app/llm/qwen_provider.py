from __future__ import annotations

import logging

import httpx

from app.core.exceptions import LLMUnavailableError
from app.llm.provider import ChatMessage, LLMResult

logger = logging.getLogger("entwin.ai.llm")


class QwenProvider:
    """OpenAI-compatible chat client for the hosted Qwen 2.5 API (DashScope)."""

    provider_name = "qwen"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        self._model = model
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[ChatMessage]) -> LLMResult:
        if not self._api_key:
            raise LLMUnavailableError("Qwen API key is not configured.")
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError("Qwen API is unreachable.") from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Qwen API request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("Qwen API request failed.") from exc

        if response.status_code >= 400:
            logger.warning("qwen_http_error status=%s model=%s", response.status_code, self._model)
            raise LLMUnavailableError("Qwen API rejected the chat request.")

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMUnavailableError("Qwen API returned an invalid response.") from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError("Qwen API returned an empty assistant message.")

        model = str(body.get("model") or self._model)
        logger.info("llm_success provider=qwen model=%s", model)
        return LLMResult(text=text.strip(), model=model, provider=self.provider_name)
