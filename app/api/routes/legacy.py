from fastapi import APIRouter, Depends

from app.api.dependencies import get_lifestyle_service, get_sleep_service
from app.schemas.lifestyle import LegacyLifestyleRequest, LegacyLifestyleResponse
from app.schemas.sleep import LegacySleepResponse, SleepRiskRequest
from app.services.lifestyle_service import LifestyleService
from app.services.sleep_service import SleepRiskService

router = APIRouter(tags=["legacy"])


@router.post("/predict/sleep_disorder", response_model=LegacySleepResponse, deprecated=True)
@router.post("/predict/sleep-disorder", response_model=LegacySleepResponse, deprecated=True)
def legacy_sleep_disorder(
    payload: SleepRiskRequest,
    service: SleepRiskService = Depends(get_sleep_service),
) -> LegacySleepResponse:
    return service.predict_legacy(payload)


@router.post("/predict/lifestyle_risk", response_model=LegacyLifestyleResponse, deprecated=True)
@router.post("/predict/lifestyle-risk", response_model=LegacyLifestyleResponse, deprecated=True)
def legacy_lifestyle_risk(
    payload: LegacyLifestyleRequest,
    service: LifestyleService = Depends(get_lifestyle_service),
) -> LegacyLifestyleResponse:
    return service.predict_from_scores(payload)
