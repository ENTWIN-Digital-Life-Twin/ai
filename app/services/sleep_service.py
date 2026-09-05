from __future__ import annotations

import logging

from app.core.constants import SLEEP_MODEL_NAME, SLEEP_RISK_DISCLAIMER
from app.core.exceptions import ModelUnavailableError, PredictionError
from app.ml.model_loader import LoadedModel
from app.ml.preprocessing.sleep import build_sleep_feature_frame, encode_sleep_row
from app.schemas.sleep import LegacySleepResponse, SleepRiskRequest, SleepRiskResponse

logger = logging.getLogger("entwin.ai.sleep")

_CLASS_TO_RISK = {
    "None": "LOW",
    "Insomnia": "MODERATE",
    "Sleep Apnea": "HIGH",
}


class SleepRiskService:
    def __init__(self, loaded: LoadedModel | None) -> None:
        self._loaded = loaded

    @property
    def loaded_model(self) -> LoadedModel:
        if self._loaded is None or self._loaded.estimator is None or not self._loaded.feature_names:
            raise ModelUnavailableError("Sleep-risk model is not loaded")
        return self._loaded

    def predict(self, request: SleepRiskRequest) -> SleepRiskResponse:
        loaded = self.loaded_model
        encoded = encode_sleep_row(
            age=request.age,
            gender=request.gender,
            sleep_duration=request.sleep_duration,
            quality_of_sleep=request.quality_of_sleep,
            physical_activity_level=request.physical_activity_level,
            stress_level=request.stress_level,
            bmi_category=request.bmi_category,
            heart_rate=request.heart_rate,
            daily_steps=request.daily_steps,
        )
        try:
            frame = build_sleep_feature_frame(encoded, loaded.feature_names)
            predicted = loaded.estimator.predict(frame)[0]
            probability = _max_predict_proba(loaded.estimator, frame)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("sleep prediction failed model=%s version=%s", loaded.name, loaded.version)
            raise PredictionError("Sleep-risk prediction failed") from exc

        predicted_class = str(predicted)
        logger.info(
            "prediction success model=%s version=%s predicted_class=%s",
            loaded.name,
            loaded.version,
            predicted_class,
        )
        return SleepRiskResponse(
            predicted_class=predicted_class,
            risk_level=_CLASS_TO_RISK.get(predicted_class, "MODERATE"),
            model_probability=probability,
            model_name=loaded.name or SLEEP_MODEL_NAME,
            model_version=loaded.version,
            disclaimer=SLEEP_RISK_DISCLAIMER,
        )

    def predict_legacy(self, request: SleepRiskRequest) -> LegacySleepResponse:
        result = self.predict(request)
        return LegacySleepResponse(
            risk=result.predicted_class,
            confidence=result.model_probability,
            predicted_class=result.predicted_class,
            risk_level=result.risk_level,
            model_probability=result.model_probability,
            model_name=result.model_name,
            model_version=result.model_version,
            disclaimer=result.disclaimer,
        )


def _max_predict_proba(estimator, frame) -> float | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    probabilities = estimator.predict_proba(frame)[0]
    return round(float(max(probabilities)), 2)
