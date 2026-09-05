from pydantic import BaseModel, Field, field_validator

from app.core.constants import SLEEP_RISK_DISCLAIMER
from app.schemas.common import APIModel

_ALLOWED_GENDERS = {"Male", "Female"}


class SleepRiskRequest(APIModel):
    age: int = Field(..., gt=0, le=120)
    gender: str
    sleep_duration: float = Field(..., ge=0, le=24)
    quality_of_sleep: int = Field(..., ge=1, le=10)
    physical_activity_level: int = Field(..., ge=0, le=200)
    stress_level: int = Field(..., ge=1, le=10)
    bmi_category: str
    heart_rate: int = Field(..., gt=0, le=220)
    daily_steps: int = Field(..., ge=0, le=50000)

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, value: object) -> object:
        if isinstance(value, str):
            titled = value.strip().title()
            if titled in _ALLOWED_GENDERS:
                return titled
        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in _ALLOWED_GENDERS:
            raise ValueError("gender must be Male or Female")
        return value

    @field_validator("bmi_category", mode="before")
    @classmethod
    def normalize_bmi(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"normal", "normal weight"}:
            return "Normal"
        if lowered == "overweight":
            return "Overweight"
        if lowered == "obese":
            return "Obese"
        return stripped

    @field_validator("bmi_category")
    @classmethod
    def validate_bmi(cls, value: str) -> str:
        if value not in {"Normal", "Overweight", "Obese"}:
            raise ValueError("bmiCategory must be Normal, Normal Weight, Overweight, or Obese")
        return value


class SleepRiskResponse(APIModel):
    predicted_class: str
    risk_level: str
    model_probability: float | None = None
    model_name: str
    model_version: str
    disclaimer: str = SLEEP_RISK_DISCLAIMER


class LegacySleepResponse(BaseModel):
    risk: str
    confidence: float | None = None
    predicted_class: str
    risk_level: str
    model_probability: float | None = None
    model_name: str
    model_version: str
    disclaimer: str = SLEEP_RISK_DISCLAIMER
