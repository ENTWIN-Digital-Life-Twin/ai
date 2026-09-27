"""Guideline-based soft labels for recommendation types."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from app.ml.recommendation.catalog import REC_TYPES, TargetDefaults

# Soft relevance in [0, 1] per type; >= LABEL_THRESHOLD → positive class.
LABEL_THRESHOLD = 0.42


def soft_relevance(
    signals: Mapping[str, float | None],
    targets: TargetDefaults | None = None,
) -> dict[str, float]:
    """Continuous need scores from common lifestyle targets (non-medical)."""
    t = targets or TargetDefaults()
    scores: dict[str, float] = {name: 0.0 for name in REC_TYPES}

    sleep = signals.get("sleep_minutes")
    if sleep is not None:
        # Full need below 6h, taper to 0 by 8h target.
        if sleep < 360:
            scores["REST"] = max(scores["REST"], 0.95)
        elif sleep < 420:
            scores["REST"] = max(scores["REST"], 0.65)
        elif sleep < t.sleep_minutes:
            scores["REST"] = max(scores["REST"], 0.35 + 0.3 * (t.sleep_minutes - sleep) / 60.0)

    hydration = signals.get("hydration_ml")
    if hydration is not None:
        if hydration < 1500:
            scores["HYDRATION"] = max(scores["HYDRATION"], 0.9)
        elif hydration < t.hydration_ml:
            gap = (t.hydration_ml - hydration) / t.hydration_ml
            scores["HYDRATION"] = max(scores["HYDRATION"], min(0.85, 0.4 + gap))

    stress = signals.get("stress_level")
    if stress is not None:
        if stress >= 8:
            scores["STRESS"] = max(scores["STRESS"], 0.95)
        elif stress >= 7:
            scores["STRESS"] = max(scores["STRESS"], 0.8)
        elif stress >= 5.5:
            scores["STRESS"] = max(scores["STRESS"], 0.5)

    fatigue = signals.get("fatigue_level")
    if fatigue is not None:
        if fatigue >= 8:
            scores["REST"] = max(scores["REST"], 0.85)
        elif fatigue >= 7:
            scores["REST"] = max(scores["REST"], 0.7)
        elif fatigue >= 5.5:
            scores["REST"] = max(scores["REST"], 0.45)

    workout = signals.get("weekly_workout_minutes")
    if workout is not None:
        if workout < 30:
            scores["ACTIVITY"] = max(scores["ACTIVITY"], 0.9)
        elif workout < 75:
            scores["ACTIVITY"] = max(scores["ACTIVITY"], 0.65)
        elif workout < t.workout_weekly_minutes:
            scores["ACTIVITY"] = max(scores["ACTIVITY"], 0.4)

    mood = signals.get("mood_level")
    if mood is not None:
        if mood <= 3:
            scores["MOOD"] = max(scores["MOOD"], 0.9)
        elif mood <= 4:
            scores["MOOD"] = max(scores["MOOD"], 0.7)
        elif mood < t.mood_comfort:
            scores["MOOD"] = max(scores["MOOD"], 0.4)

    steps = signals.get("daily_steps")
    if steps is not None:
        if steps < 4000:
            scores["ACTIVITY"] = max(scores["ACTIVITY"], 0.75)
        elif steps < t.steps:
            scores["ACTIVITY"] = max(scores["ACTIVITY"], 0.45)

    # Interaction boosts: combinations matter more than isolated signals.
    if sleep is not None and stress is not None and sleep < 420 and stress >= 6:
        scores["REST"] = min(1.0, scores["REST"] + 0.12)
        scores["STRESS"] = min(1.0, scores["STRESS"] + 0.1)
    if fatigue is not None and sleep is not None and fatigue >= 6 and sleep < 450:
        scores["REST"] = min(1.0, scores["REST"] + 0.1)
    if mood is not None and stress is not None and mood <= 5 and stress >= 6:
        scores["MOOD"] = min(1.0, scores["MOOD"] + 0.1)
        scores["STRESS"] = min(1.0, scores["STRESS"] + 0.08)

    return scores


def multi_hot(scores: Mapping[str, float], threshold: float = LABEL_THRESHOLD) -> np.ndarray:
    return np.asarray(
        [1 if scores.get(name, 0.0) >= threshold else 0 for name in REC_TYPES],
        dtype=np.int32,
    )


def soft_vector(scores: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(scores.get(name, 0.0)) for name in REC_TYPES], dtype=np.float64)
