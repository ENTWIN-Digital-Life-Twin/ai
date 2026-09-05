from pydantic import Field

from app.schemas.common import APIModel


class RecommendationRequest(APIModel):
    sleep_minutes: float | None = Field(default=None, ge=0, le=1440)
    hydration_ml: float | None = Field(default=None, ge=0, le=20000)
    stress_level: float | None = Field(default=None, ge=1, le=10)
    fatigue_level: float | None = Field(default=None, ge=1, le=10)
    weekly_workout_minutes: float | None = Field(default=None, ge=0, le=10080)
    mood_level: float | None = Field(default=None, ge=1, le=10)
    daily_steps: float | None = Field(default=None, ge=0, le=100000)


class RecommendationItem(APIModel):
    type: str
    priority: str
    message: str


class RecommendationResponse(APIModel):
    recommendations: list[RecommendationItem]
    engine: str = "RULE_BASED_BASELINE"
