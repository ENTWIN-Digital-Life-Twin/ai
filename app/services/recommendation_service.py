from __future__ import annotations

import logging

from app.core.constants import ENGINE_RULE_BASED_BASELINE
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
    """Transparent threshold rules. Not medical advice and not an LLM."""

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
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
