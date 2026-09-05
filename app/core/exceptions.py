from typing import Any


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, error: str = "Internal Server Error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error = error


class DomainValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, error="Bad Request")


class ModelLoadError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503, error="Service Unavailable")


class ModelUnavailableError(AppError):
    def __init__(self, message: str = "Required model is unavailable") -> None:
        super().__init__(message, status_code=503, error="Service Unavailable")


class PredictionError(AppError):
    def __init__(self, message: str = "Prediction failed") -> None:
        super().__init__(message, status_code=500, error="Internal Server Error")


def error_body(status: int, error: str, message: str, path: str, field_errors: dict[str, Any] | None = None) -> dict:
    body: dict[str, Any] = {
        "status": status,
        "error": error,
        "message": message,
        "path": path,
    }
    if field_errors:
        body["fieldErrors"] = field_errors
    return body
