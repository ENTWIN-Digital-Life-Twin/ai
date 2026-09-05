from fastapi import APIRouter, Depends

from app.api.dependencies import get_lifestyle_service
from app.schemas.lifestyle import LifestyleRiskResponse, WellnessLifestyleRequest
from app.services.lifestyle_service import LifestyleService

router = APIRouter(tags=["lifestyle"])


@router.post("/lifestyle-risk", response_model=LifestyleRiskResponse)
def lifestyle_risk(
    payload: WellnessLifestyleRequest,
    service: LifestyleService = Depends(get_lifestyle_service),
) -> LifestyleRiskResponse:
    return service.predict_from_wellness(payload)
