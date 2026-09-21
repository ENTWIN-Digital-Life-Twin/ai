from typing import Any

from pydantic import Field, field_validator

from app.schemas.common import APIModel


class ChatTurn(APIModel):
    role: str = Field(min_length=1, max_length=16)
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(APIModel):
    question: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any] | None = None
    history: list[ChatTurn] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    @field_validator("history")
    @classmethod
    def cap_history(cls, value: list[ChatTurn]) -> list[ChatTurn]:
        cleaned: list[ChatTurn] = []
        for turn in value[-20:]:
            role = turn.role.strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = turn.content.strip()
            if content:
                cleaned.append(ChatTurn(role=role, content=content[:2000]))
        return cleaned[-12:]


class ProposedTask(APIModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    duration_minutes: int = Field(default=45, ge=5, le=24 * 60)
    priority: str = "MEDIUM"
    category: str = "WORK"


class ChatResponse(APIModel):
    answer: str
    engine: str
    provider: str
    model: str
    proposed_action: str | None = None
    proposed_task: ProposedTask | None = None
    disclaimer: str
