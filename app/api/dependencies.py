from fastapi import Request

from app.core.exceptions import ModelUnavailableError
from app.llm.provider import LLMProvider
from app.ml.model_loader import ModelRegistry
from app.services.chat_service import ChatService
from app.services.lifestyle_service import LifestyleService
from app.services.recommendation_service import RecommendationService
from app.services.sleep_service import SleepRiskService
from app.services.task_duration_service import TaskDurationService


def get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "models", None)
    if registry is None:
        raise ModelUnavailableError("Model registry is not initialized")
    return registry


def get_sleep_service(request: Request) -> SleepRiskService:
    return SleepRiskService(get_registry(request).sleep)


def get_lifestyle_service(request: Request) -> LifestyleService:
    return LifestyleService(request.app.state.settings)


def get_task_duration_service() -> TaskDurationService:
    return TaskDurationService()


def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


def get_llm_provider(request: Request) -> LLMProvider | None:
    return getattr(request.app.state, "llm", None)


def get_chat_service(request: Request) -> ChatService:
    return ChatService(get_llm_provider(request))
