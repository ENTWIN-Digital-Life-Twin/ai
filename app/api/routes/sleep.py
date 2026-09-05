from fastapi import APIRouter, Depends

from app.api.dependencies import get_sleep_service
from app.schemas.sleep import SleepRiskRequest, SleepRiskResponse
from app.services.sleep_service import SleepRiskService

router = APIRouter(tags=["sleep"])


@router.post("/sleep-risk", response_model=SleepRiskResponse)
def sleep_risk(
    payload: SleepRiskRequest,
    service: SleepRiskService = Depends(get_sleep_service),
) -> SleepRiskResponse:
    return service.predict(payload)
