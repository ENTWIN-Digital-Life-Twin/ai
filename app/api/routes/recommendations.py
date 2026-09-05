from fastapi import APIRouter, Depends

from app.api.dependencies import get_recommendation_service
from app.schemas.recommendations import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(tags=["recommendations"])


@router.post("/recommendations", response_model=RecommendationResponse)
def recommendations(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    return service.recommend(payload)
