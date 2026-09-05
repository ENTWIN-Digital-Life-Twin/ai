from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import joblib

from app.core.config import Settings
from app.core.constants import (
    LIFESTYLE_MODEL_NAME,
    LIFESTYLE_MODEL_VERSION,
    SLEEP_MODEL_NAME,
    SLEEP_MODEL_VERSION,
)
from app.core.exceptions import ModelLoadError
from app.ml.metadata import load_json_metadata, public_model_info

logger = logging.getLogger("entwin.ai.models")


@dataclass
class LoadedModel:
    name: str
    version: str
    estimator: Any
    feature_names: list[str] | None
    algorithm: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    used_in_production: bool = True

    @property
    def feature_count(self) -> int | None:
        if self.feature_names is not None:
            return len(self.feature_names)
        n_features = getattr(self.estimator, "n_features_in_", None)
        return int(n_features) if n_features is not None else None


@dataclass
class ModelRegistry:
    sleep: LoadedModel | None = None
    lifestyle: LoadedModel | None = None

    def public_catalog(self) -> list[dict[str, Any]]:
        models = []
        if self.sleep is not None:
            models.append(
                public_model_info(
                    name=self.sleep.name,
                    version=self.sleep.version,
                    algorithm=self.sleep.algorithm,
                    loaded=True,
                    feature_count=self.sleep.feature_count,
                    used_in_production=self.sleep.used_in_production,
                )
            )
        else:
            models.append(
                public_model_info(
                    name=SLEEP_MODEL_NAME,
                    version=SLEEP_MODEL_VERSION,
                    algorithm=None,
                    loaded=False,
                    feature_count=None,
                    used_in_production=True,
                )
            )
        if self.lifestyle is not None:
            models.append(
                public_model_info(
                    name=self.lifestyle.name,
                    version=self.lifestyle.version,
                    algorithm=self.lifestyle.algorithm,
                    loaded=True,
                    feature_count=self.lifestyle.feature_count,
                    used_in_production=False,
                    notes=(
                        "Trained on synthetic 0-100 scores labeled by the rule engine; "
                        "production lifestyle-risk uses RULE_BASED_BASELINE."
                    ),
                )
            )
        else:
            models.append(
                public_model_info(
                    name=LIFESTYLE_MODEL_NAME,
                    version=LIFESTYLE_MODEL_VERSION,
                    algorithm=None,
                    loaded=False,
                    feature_count=None,
                    used_in_production=False,
                    notes="Optional artifact; production lifestyle-risk uses RULE_BASED_BASELINE.",
                )
            )
        return models


def load_registry(settings: Settings) -> ModelRegistry:
    sleep = _load_sleep(settings)
    lifestyle = _load_lifestyle(settings)
    return ModelRegistry(sleep=sleep, lifestyle=lifestyle)


def _load_sleep(settings: Settings) -> LoadedModel | None:
    model_path = settings.resolve_path(settings.sleep_model_path)
    features_path = settings.resolve_path(settings.sleep_features_path)
    metadata_path = settings.resolve_path(settings.sleep_metadata_path)

    if not model_path.is_file():
        message = f"Required sleep model file is missing: {model_path.name}"
        if settings.sleep_model_required:
            raise ModelLoadError(message)
        logger.warning(message)
        return None

    if not features_path.is_file():
        message = f"Required sleep feature-order file is missing: {features_path.name}"
        if settings.sleep_model_required:
            raise ModelLoadError(message)
        logger.warning(message)
        return None

    try:
        estimator = joblib.load(model_path)
        feature_names = joblib.load(features_path)
    except Exception as exc:  # noqa: BLE001 - startup must fail clearly
        raise ModelLoadError("Failed to load the sleep-risk model artifacts.") from exc

    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise ModelLoadError("Sleep feature metadata must be a list of feature name strings.")

    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(feature_names):
        raise ModelLoadError(
            "Sleep feature file does not match the loaded model "
            f"(model expects {n_features} features, file has {len(feature_names)})."
        )

    model_feature_names = getattr(estimator, "feature_names_in_", None)
    if model_feature_names is not None and list(model_feature_names) != feature_names:
        raise ModelLoadError("Sleep feature order does not match the serialized model's feature_names_in_.")

    metadata = load_json_metadata(metadata_path)
    version = str(metadata.get("version") or SLEEP_MODEL_VERSION)
    algorithm = _algorithm_name(estimator, metadata)
    logger.info("loaded model=%s version=%s algorithm=%s features=%s", SLEEP_MODEL_NAME, version, algorithm, len(feature_names))
    return LoadedModel(
        name=SLEEP_MODEL_NAME,
        version=version,
        estimator=estimator,
        feature_names=feature_names,
        algorithm=algorithm,
        metadata=metadata,
        used_in_production=True,
    )


def _load_lifestyle(settings: Settings) -> LoadedModel | None:
    model_path = settings.resolve_path(settings.lifestyle_model_path)
    metadata_path = settings.resolve_path(settings.lifestyle_metadata_path)
    if not model_path.is_file():
        message = f"Optional lifestyle model file is missing: {model_path.name}"
        if settings.lifestyle_model_required:
            raise ModelLoadError(message)
        logger.info(message)
        return None

    try:
        estimator = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        if settings.lifestyle_model_required:
            raise ModelLoadError("Failed to load the lifestyle model artifact.") from exc
        logger.warning("optional lifestyle model could not be loaded: %s", type(exc).__name__)
        return None

    metadata = load_json_metadata(metadata_path)
    version = str(metadata.get("version") or LIFESTYLE_MODEL_VERSION)
    algorithm = _algorithm_name(estimator, metadata)
    raw_names = getattr(estimator, "feature_names_in_", None)
    feature_names = list(raw_names) if raw_names is not None else None
    logger.info(
        "loaded optional model=%s version=%s algorithm=%s used_in_production=false",
        LIFESTYLE_MODEL_NAME,
        version,
        algorithm,
    )
    return LoadedModel(
        name=LIFESTYLE_MODEL_NAME,
        version=version,
        estimator=estimator,
        feature_names=feature_names,
        algorithm=algorithm,
        metadata=metadata,
        used_in_production=False,
    )


def _algorithm_name(estimator: Any, metadata: dict[str, Any]) -> str | None:
    if metadata.get("algorithm"):
        return str(metadata["algorithm"])
    return type(estimator).__name__ if estimator is not None else None
