from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/ai/health", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="UP", service=settings.app_name, version=settings.app_version)
