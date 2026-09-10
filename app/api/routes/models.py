from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_registry
from app.llm.factory import llm_public_info
from app.ml.model_loader import ModelRegistry
from app.schemas.health import ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models(request: Request, registry: ModelRegistry = Depends(get_registry)) -> ModelsResponse:
    catalog = registry.public_catalog()
    catalog.append(llm_public_info(request.app.state.settings, getattr(request.app.state, "llm", None)))
    return ModelsResponse(models=catalog)
