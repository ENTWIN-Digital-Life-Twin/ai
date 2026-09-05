from __future__ import annotations

import logging
import math

from app.core.constants import ENGINE_BASELINE_ESTIMATOR
from app.schemas.planning import TaskDurationRequest, TaskDurationResponse

logger = logging.getLogger("entwin.ai.planning")

CATEGORY_BASE_MINUTES = {
    "STUDY": 90,
    "WORK": 60,
    "SPORT": 45,
    "EXERCISE": 45,
    "WORKOUT": 45,
    "CHORES": 30,
    "PERSONAL": 40,
    "HEALTH": 30,
    "OTHER": 45,
}

COMPLEXITY_FACTOR = {
    "EASY": 0.8,
    "MEDIUM": 1.0,
    "HARD": 1.3,
}

ENERGY_FACTOR = {
    "LOW": 0.9,
    "MEDIUM": 1.0,
    "HIGH": 1.15,
}


class TaskDurationService:
    """Deterministic duration estimator. This is not a trained ML model."""

    def predict(self, request: TaskDurationRequest) -> TaskDurationResponse:
        complexity = request.complexity or "MEDIUM"
        energy = request.energy_required or "MEDIUM"
        complexity_mult = COMPLEXITY_FACTOR[complexity]
        energy_mult = ENERGY_FACTOR[energy]

        if request.user_estimate_minutes is not None and request.historical_average_minutes is not None:
            base = 0.6 * request.user_estimate_minutes + 0.4 * request.historical_average_minutes
            minutes = base * (1.0 + 0.5 * (complexity_mult - 1.0) + 0.25 * (energy_mult - 1.0))
        elif request.user_estimate_minutes is not None:
            minutes = request.user_estimate_minutes * (
                1.0 + 0.5 * (complexity_mult - 1.0) + 0.25 * (energy_mult - 1.0)
            )
        elif request.historical_average_minutes is not None:
            minutes = request.historical_average_minutes * (
                1.0 + 0.5 * (complexity_mult - 1.0) + 0.25 * (energy_mult - 1.0)
            )
        else:
            category = request.category or "OTHER"
            base = CATEGORY_BASE_MINUTES.get(category, CATEGORY_BASE_MINUTES["OTHER"])
            minutes = base * complexity_mult * energy_mult

        predicted = max(5, int(math.floor(minutes + 0.5)))
        logger.info("prediction success model=task-duration engine=%s", ENGINE_BASELINE_ESTIMATOR)
        return TaskDurationResponse(
            predicted_duration_minutes=predicted,
            engine=ENGINE_BASELINE_ESTIMATOR,
            confidence=None,
        )
