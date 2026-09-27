"""Feature engineering for the recommendation ranker."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.ml.recommendation.catalog import TargetDefaults

FEATURE_NAMES: list[str] = [
    "sleep_minutes",
    "hydration_ml",
    "stress_level",
    "fatigue_level",
    "weekly_workout_minutes",
    "mood_level",
    "daily_steps",
    "sleep_deficit",
    "hydration_deficit",
    "stress_excess",
    "fatigue_excess",
    "workout_deficit",
    "mood_deficit",
    "steps_deficit",
    "stress_x_sleep_deficit",
    "fatigue_x_sleep_deficit",
    "stress_x_mood_deficit",
    "activity_x_steps_deficit",
    "signal_count",
    "missing_fraction",
]


def _get(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                return None
    return None


def signals_from_request(request: Any) -> dict[str, float | None]:
    """Normalize a RecommendationRequest (or mapping) into signal dict."""
    if isinstance(request, Mapping):
        return {
            "sleep_minutes": _get(request, "sleep_minutes", "sleepMinutes"),
            "hydration_ml": _get(request, "hydration_ml", "hydrationMl"),
            "stress_level": _get(request, "stress_level", "stressLevel"),
            "fatigue_level": _get(request, "fatigue_level", "fatigueLevel"),
            "weekly_workout_minutes": _get(
                request, "weekly_workout_minutes", "weeklyWorkoutMinutes"
            ),
            "mood_level": _get(request, "mood_level", "moodLevel"),
            "daily_steps": _get(request, "daily_steps", "dailySteps"),
        }
    return {
        "sleep_minutes": getattr(request, "sleep_minutes", None),
        "hydration_ml": getattr(request, "hydration_ml", None),
        "stress_level": getattr(request, "stress_level", None),
        "fatigue_level": getattr(request, "fatigue_level", None),
        "weekly_workout_minutes": getattr(request, "weekly_workout_minutes", None),
        "mood_level": getattr(request, "mood_level", None),
        "daily_steps": getattr(request, "daily_steps", None),
    }


def _deficit(value: float | None, target: float, *, higher_is_worse: bool = False) -> float:
    if value is None:
        return 0.0
    if higher_is_worse:
        return max(0.0, float(value) - target)
    return max(0.0, target - float(value))


def build_feature_vector(
    signals: Mapping[str, float | None],
    targets: TargetDefaults | None = None,
) -> np.ndarray:
    """Return a 1-D float array aligned with FEATURE_NAMES.

    Missing raw signals are imputed to neutral target values so sparse API
    requests (e.g. only stressLevel) do not spuriously trigger other types.
    Deficits stay zero when the signal was not provided.
    """
    t = targets or TargetDefaults()
    sleep = signals.get("sleep_minutes")
    hydration = signals.get("hydration_ml")
    stress = signals.get("stress_level")
    fatigue = signals.get("fatigue_level")
    workout = signals.get("weekly_workout_minutes")
    mood = signals.get("mood_level")
    steps = signals.get("daily_steps")

    raw = [sleep, hydration, stress, fatigue, workout, mood, steps]
    present = sum(1 for v in raw if v is not None)
    missing_fraction = 1.0 - (present / len(raw))

    sleep_deficit = _deficit(sleep, t.sleep_minutes)
    hydration_deficit = _deficit(hydration, t.hydration_ml)
    stress_excess = _deficit(stress, t.stress_comfort, higher_is_worse=True)
    fatigue_excess = _deficit(fatigue, t.fatigue_comfort, higher_is_worse=True)
    workout_deficit = _deficit(workout, t.workout_weekly_minutes)
    mood_deficit = _deficit(mood, t.mood_comfort)
    steps_deficit = _deficit(steps, t.steps)

    # Neutral imputation for tree splits (not used for deficit calculation above).
    sleep_f = float(sleep) if sleep is not None else t.sleep_minutes
    hydration_f = float(hydration) if hydration is not None else t.hydration_ml
    stress_f = float(stress) if stress is not None else t.stress_comfort
    fatigue_f = float(fatigue) if fatigue is not None else t.fatigue_comfort
    workout_f = float(workout) if workout is not None else t.workout_weekly_minutes
    mood_f = float(mood) if mood is not None else t.mood_comfort
    steps_f = float(steps) if steps is not None else t.steps

    values = [
        sleep_f,
        hydration_f,
        stress_f,
        fatigue_f,
        workout_f,
        mood_f,
        steps_f,
        sleep_deficit,
        hydration_deficit,
        stress_excess,
        fatigue_excess,
        workout_deficit,
        mood_deficit,
        steps_deficit,
        stress_excess * (sleep_deficit / max(t.sleep_minutes, 1.0)),
        fatigue_excess * (sleep_deficit / max(t.sleep_minutes, 1.0)),
        stress_excess * (mood_deficit / 10.0),
        (workout_deficit / max(t.workout_weekly_minutes, 1.0))
        * (steps_deficit / max(t.steps, 1.0)),
        float(present),
        missing_fraction,
    ]
    return np.asarray(values, dtype=np.float64)


def build_feature_matrix(
    signal_rows: list[Mapping[str, float | None]],
    targets: TargetDefaults | None = None,
) -> np.ndarray:
    return np.vstack([build_feature_vector(row, targets) for row in signal_rows])
