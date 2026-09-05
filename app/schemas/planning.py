from pydantic import Field, field_validator

from app.schemas.common import APIModel

_COMPLEXITY = {"EASY", "MEDIUM", "HARD"}
_ENERGY = {"LOW", "MEDIUM", "HIGH"}


class TaskDurationRequest(APIModel):
    category: str | None = Field(default=None, max_length=100)
    complexity: str | None = None
    energy_required: str | None = None
    user_estimate_minutes: int | None = Field(default=None, gt=0, le=24 * 60)
    historical_average_minutes: int | None = Field(default=None, gt=0, le=24 * 60)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped.upper() if stripped else None
        return value

    @field_validator("complexity", mode="before")
    @classmethod
    def normalize_complexity(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, value: str | None) -> str | None:
        if value is not None and value not in _COMPLEXITY:
            raise ValueError("complexity must be EASY, MEDIUM, or HARD")
        return value

    @field_validator("energy_required", mode="before")
    @classmethod
    def normalize_energy(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("energy_required")
    @classmethod
    def validate_energy(cls, value: str | None) -> str | None:
        if value is not None and value not in _ENERGY:
            raise ValueError("energyRequired must be LOW, MEDIUM, or HIGH")
        return value


class TaskDurationResponse(APIModel):
    predicted_duration_minutes: int
    engine: str
    confidence: float | None = None
