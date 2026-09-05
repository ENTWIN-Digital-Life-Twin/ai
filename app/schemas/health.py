from app.schemas.common import APIModel


class HealthResponse(APIModel):
    status: str
    service: str
    version: str


class ModelsResponse(APIModel):
    models: list[dict]
