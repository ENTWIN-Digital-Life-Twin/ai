from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import APIModel


class WellnessLifestyleRequest(APIModel):
    """Normalized lifestyle signals compatible with Wellness weekly summaries.

    All fields are optional. Missing values are skipped; they are not treated as zero.
    At least one signal is required.
    """

    average_sleep_minutes: float | None = Field(default=None, ge=0, le=1440)
    average_hydration_ml: float | None = Field(default=None, ge=0, le=20000)
    weekly_workout_minutes: float | None = Field(default=None, ge=0, le=10080)
    average_stress: float | None = Field(default=None, ge=1, le=10)
    average_fatigue: float | None = Field(default=None, ge=1, le=10)
    average_mood: float | None = Field(default=None, ge=1, le=10)
    average_daily_steps: float | None = Field(default=None, ge=0, le=100000)


class LegacyLifestyleRequest(BaseModel):
    """Original 0-100 score contract. Snake_case is preserved for teammate clients."""

    model_config = ConfigDict(populate_by_name=True)
    sleep_score: float = Field(..., ge=0, le=100)
    hydration_score: float = Field(..., ge=0, le=100)
    activity_score: float = Field(..., ge=0, le=100)
    stress_score: float = Field(..., ge=0, le=100)


class LifestyleRiskResponse(APIModel):
    risk_level: str
    score: float | None = None
    engine: str
    factors: list[str]
    model_version: str | None = None


class LegacyLifestyleResponse(BaseModel):
    risk: str
    confidence: float
    contributing_factors: list[str]
    engine: str
    risk_level: str
    score: float | None = None
    factors: list[str]
