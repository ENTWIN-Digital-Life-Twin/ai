from typing import Any

from pydantic import Field, field_validator

from app.schemas.common import APIModel


class ChatRequest(APIModel):
    question: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any] | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class ChatResponse(APIModel):
    answer: str
    engine: str
    provider: str
    model: str
    proposed_action: str | None = None
    disclaimer: str
