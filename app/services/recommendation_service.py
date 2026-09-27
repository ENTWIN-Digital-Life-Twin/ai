from __future__ import annotations

import logging
from typing import Any

from app.core.constants import ENGINE_ML_RECOMMENDER, ENGINE_RULE_BASED_BASELINE
from app.ml.model_loader import LoadedModel
from app.ml.recommendation.features import build_feature_vector, signals_from_request
from app.ml.recommendation.ranker import rank_recommendations
from app.schemas.recommendations import RecommendationItem, RecommendationRequest, RecommendationResponse

logger = logging.getLogger("entwin.ai.recommendations")

SLEEP_HIGH_PRIORITY_MINUTES = 360.0
SLEEP_MEDIUM_PRIORITY_MINUTES = 420.0
HYDRATION_HIGH_ML = 1500.0
STRESS_HIGH = 7.0
FATIGUE_HIGH = 7.0
WORKOUT_HIGH_PRIORITY_MINUTES = 30.0
WORKOUT_MEDIUM_PRIORITY_MINUTES = 75.0
MOOD_LOW = 4.0
STEPS_LOW = 4000.0


class RecommendationService:
    """ML multi-label ranker with transparent rule fallback. Not medical advice."""

    def __init__(self, recommendation_model: LoadedModel | None = None) -> None:
        self._model = recommendation_model

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        if self._model is not None and self._model.estimator is not None:
            try:
                return self._recommend_ml(request)
            except Exception:  # noqa: BLE001 - never fail the endpoint; fall back to rules
                logger.exception("ml recommender failed; falling back to rules")

        return self._recommend_rules(request)

    def _recommend_ml(self, request: RecommendationRequest) -> RecommendationResponse:
        signals = signals_from_request(request)
        vector = build_feature_vector(signals).reshape(1, -1)
        estimator = self._model.estimator
        feature_names = self._model.feature_names
        if feature_names and len(feature_names) != vector.shape[1]:
            raise ValueError(
                f"feature count mismatch: model expects {len(feature_names)}, got {vector.shape[1]}"
            )

        probabilities = self._predict_probabilities(estimator, vector)
        ranked = rank_recommendations(probabilities)
        items = [
            RecommendationItem(
                type=item.type,
                priority=item.priority,
                message=item.message,
                score=item.score,
                confidence=item.confidence,
            )
            for item in ranked
        ]
        logger.info(
            "prediction success model=recommendations engine=%s count=%s",
            ENGINE_ML_RECOMMENDER,
            len(items),
        )
        return RecommendationResponse(recommendations=items, engine=ENGINE_ML_RECOMMENDER)

    @staticmethod
    def _predict_probabilities(estimator: Any, vector) -> dict[str, float]:
        from app.ml.recommendation.catalog import REC_TYPES

        raw = estimator.predict_proba(vector)
        probs: dict[str, float] = {}
        if isinstance(raw, list):
            for name, arr in zip(REC_TYPES, raw):
                if getattr(arr, "ndim", 1) == 2 and arr.shape[1] == 2:
                    probs[name] = float(arr[0, 1])
                else:
                    probs[name] = float(arr[0])
        else:
            row = raw[0]
            for i, name in enumerate(REC_TYPES):
                probs[name] = float(row[i]) if i < len(row) else 0.0
        return probs

    def _recommend_rules(self, request: RecommendationRequest) -> RecommendationResponse:
        items: list[RecommendationItem] = []

        if request.hydration_ml is not None and request.hydration_ml < HYDRATION_HIGH_ML:
            items.append(
                RecommendationItem(
                    type="HYDRATION",
                    priority="HIGH",
                    message=(
                        "Your recent hydration is below a common daily intake target. "
                        "You may want to drink water more regularly during the day."
                    ),
                )
            )

        if request.sleep_minutes is not None:
            if request.sleep_minutes < SLEEP_HIGH_PRIORITY_MINUTES:
                items.append(
                    RecommendationItem(
                        type="REST",
                        priority="HIGH",
                        message=(
                            "Your recent sleep duration is well below a common nightly target. "
                            "Consider protecting a longer sleep window."
                        ),
                    )
                )
            elif request.sleep_minutes < SLEEP_MEDIUM_PRIORITY_MINUTES:
                items.append(
                    RecommendationItem(
                        type="REST",
                        priority="MEDIUM",
                        message=(
                            "Your recent sleep duration is below a common nightly target. "
                            "You may want to keep a more consistent bedtime routine."
                        ),
                    )
                )

        if request.stress_level is not None and request.stress_level >= STRESS_HIGH:
            items.append(
                RecommendationItem(
                    type="STRESS",
                    priority="HIGH",
                    message=(
                        "Your recent pattern shows elevated stress. "
                        "You may want to schedule lighter tasks and short recovery breaks."
                    ),
                )
            )

        if request.fatigue_level is not None and request.fatigue_level >= FATIGUE_HIGH:
            items.append(
                RecommendationItem(
                    type="REST",
                    priority="MEDIUM",
                    message=(
                        "Your recent pattern shows high fatigue. "
                        "Consider prioritizing rest and reducing dense work blocks."
                    ),
                )
            )

        if request.weekly_workout_minutes is not None:
            if request.weekly_workout_minutes < WORKOUT_HIGH_PRIORITY_MINUTES:
                items.append(
                    RecommendationItem(
                        type="ACTIVITY",
                        priority="HIGH",
                        message=(
                            "Your recent activity minutes are well below a common weekly movement target. "
                            "You may want to add light movement you enjoy."
                        ),
                    )
                )
            elif request.weekly_workout_minutes < WORKOUT_MEDIUM_PRIORITY_MINUTES:
                items.append(
                    RecommendationItem(
                        type="ACTIVITY",
                        priority="MEDIUM",
                        message=(
                            "Your recent activity minutes are below a common weekly movement target. "
                            "Consider adding a few short walking or mobility sessions."
                        ),
                    )
                )

        if request.mood_level is not None and request.mood_level <= MOOD_LOW:
            items.append(
                RecommendationItem(
                    type="MOOD",
                    priority="MEDIUM",
                    message=(
                        "Your recent mood ratings are on the low side. "
                        "You may want to keep lighter plans and include activities you usually enjoy."
                    ),
                )
            )

        if request.daily_steps is not None and request.daily_steps < STEPS_LOW:
            items.append(
                RecommendationItem(
                    type="ACTIVITY",
                    priority="MEDIUM",
                    message=(
                        "Your recent step count is below a common daily walking range. "
                        "Consider adding short walks if that fits your day."
                    ),
                )
            )

        logger.info(
            "prediction success model=recommendations engine=%s count=%s",
            ENGINE_RULE_BASED_BASELINE,
            len(items),
        )
        return RecommendationResponse(recommendations=items, engine=ENGINE_RULE_BASED_BASELINE)
