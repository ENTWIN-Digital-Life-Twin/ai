from pydantic import BaseModel, ConfigDict


def to_camel(snake: str) -> str:
    parts = snake.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


class APIModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )
