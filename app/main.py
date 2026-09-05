from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import health, legacy, lifestyle, models, planning, recommendations, sleep
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, ModelLoadError, error_body
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.ml.model_loader import load_registry

logger = logging.getLogger("entwin.ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    app.state.settings = settings
    try:
        app.state.models = load_registry(settings)
    except ModelLoadError:
        logger.exception("required model failed to load at startup")
        raise
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    application = FastAPI(
        title="ENTWIN AI Service",
        description=(
            "Lifestyle and sleep risk indicators for ENTWIN Digital Life Twin. "
            "Outputs are informational and are not medical diagnoses."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )
    application.add_middleware(RequestLoggingMiddleware)
    _register_exception_handlers(application)

    application.include_router(health.router)
    application.include_router(models.router, prefix="/api/v1/ai")
    application.include_router(sleep.router, prefix="/api/v1/ai")
    application.include_router(lifestyle.router, prefix="/api/v1/ai")
    application.include_router(planning.router, prefix="/api/v1/ai")
    application.include_router(recommendations.router, prefix="/api/v1/ai")
    application.include_router(legacy.router)

    @application.get("/")
    def root() -> dict[str, str]:
        return {"message": "AI service is running", "service": settings.app_name, "version": settings.app_version}

    return application


def _register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error status=%s path=%s", exc.status_code, request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.status_code, exc.error, exc.message, request.url.path),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        field_errors: dict[str, str] = {}
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", []) if part != "body")
            field_errors[location or "request"] = err.get("msg", "Invalid value")
        return JSONResponse(
            status_code=422,
            content=error_body(
                422,
                "Unprocessable Entity",
                "Request validation failed",
                request.url.path,
                field_errors,
            ),
        )

    @application.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, (HTTPException, StarletteHTTPException, RequestValidationError, AppError)):
            raise exc
        logger.exception("unhandled_error path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(
                500,
                "Internal Server Error",
                "An unexpected error occurred",
                request.url.path,
            ),
        )


app = create_app()
