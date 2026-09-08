from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.core.constants import ASSISTANT_NAME
from app.llm.ollama_provider import OllamaProvider
from app.llm.provider import LLMProvider
from app.ml.metadata import public_model_info

logger = logging.getLogger("entwin.ai.llm")


def create_llm_provider(settings: Settings) -> LLMProvider | None:
    provider_name = (settings.llm_provider or "").strip().lower()
    if provider_name in {"", "none", "off", "disabled"}:
        logger.info("llm_provider disabled")
        return None
    if provider_name != "ollama":
        logger.warning("unknown llm_provider=%s; assistant disabled", provider_name)
        return None
    if not settings.ollama_base_url.strip() or not settings.ollama_model.strip():
        logger.warning("ollama configuration incomplete; assistant disabled")
        return None
    logger.info(
        "llm_provider configured provider=ollama model=%s base_host_configured=true",
        settings.ollama_model,
    )
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        connect_timeout_seconds=settings.ollama_connect_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def llm_public_info(settings: Settings, provider: LLMProvider | None) -> dict[str, Any]:
    algorithm = settings.ollama_model if (settings.llm_provider or "").strip().lower() == "ollama" else settings.llm_provider
    return public_model_info(
        name=ASSISTANT_NAME,
        version=settings.app_version,
        algorithm=algorithm or None,
        loaded=provider is not None,
        feature_count=None,
        used_in_production=True,
        notes=(
            "Remote LLMProvider (Ollama HTTP). Uses caller-supplied context only; "
            "not a source of business truth and not a medical diagnosis."
        ),
    )
