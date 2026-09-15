from typing import Literal

from pydantic import Field, field_validator

from app.schemas.common import APIModel

FormType = Literal["TASK", "EVENT", "MEAL", "WORKOUT", "WELLNESS"]


class FormSuggestRequest(APIModel):
    form_type: FormType
    title: str | None = Field(default=None, max_length=300)
    category: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    locale: str | None = Field(default="en", max_length=10)

    @field_validator("form_type", "category", mode="before")
    @classmethod
    def normalize_upper(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped.upper() if stripped else value
        return value


class FormSuggestion(APIModel):
    field: str
    value: str
    label: str
    reason: str | None = None


class FormSuggestResponse(APIModel):
    suggestions: list[FormSuggestion]
    engine: str
