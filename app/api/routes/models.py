from fastapi import APIRouter, Depends

from app.api.dependencies import get_registry
from app.ml.model_loader import ModelRegistry
from app.schemas.health import ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelsResponse:
    return ModelsResponse(models=registry.public_catalog())
