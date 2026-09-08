from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "entwin-ai-service"
    app_version: str = "0.1.0"
    ai_service_host: str = "0.0.0.0"
    ai_service_port: int = 8090
    ai_log_level: str = "INFO"

    sleep_model_path: str = "app/models/sleep_disorder_model.joblib"
    sleep_features_path: str = "app/models/sleep_disorder_features.joblib"
    sleep_metadata_path: str = "app/models/sleep_disorder_model.metadata.json"
    lifestyle_model_path: str = "app/models/lifestyle_risk_model.joblib"
    lifestyle_metadata_path: str = "app/models/lifestyle_risk_model.metadata.json"

    sleep_model_required: bool = True
    lifestyle_model_required: bool = False

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    llm_max_tokens: int = Field(default=512, gt=0, le=4096)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    sleep_target_minutes: float = 480.0
    hydration_target_ml: float = 2000.0
    workout_target_weekly_minutes: float = 150.0
    steps_target: float = 8000.0
    lifestyle_score_threshold: float = 60.0

    def resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
