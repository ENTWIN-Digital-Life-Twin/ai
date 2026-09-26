"""Recommendation catalog: types, targets, templates, severity weights."""

from __future__ import annotations

from dataclasses import dataclass

REC_TYPES = ("HYDRATION", "REST", "STRESS", "ACTIVITY", "MOOD")

# Severity multipliers used at ranking time (higher = more urgent when triggered).
SEVERITY_WEIGHT: dict[str, float] = {
    "HYDRATION": 1.05,
    "REST": 1.2,
    "STRESS": 1.15,
    "ACTIVITY": 1.0,
    "MOOD": 0.95,
}

# Score floors for priority bands after P(type) * severity.
PRIORITY_HIGH = 0.72
PRIORITY_MEDIUM = 0.45
# Below this calibrated floor → omit recommendation (healthy profiles stay empty).
SCORE_FLOOR = 0.38

TOP_K = 3


@dataclass(frozen=True)
class TargetDefaults:
    sleep_minutes: float = 480.0
    hydration_ml: float = 2000.0
    workout_weekly_minutes: float = 150.0
    steps: float = 8000.0
    stress_comfort: float = 4.0
    fatigue_comfort: float = 4.0
    mood_comfort: float = 6.0


MESSAGES: dict[str, dict[str, str]] = {
    "HYDRATION": {
        "HIGH": (
            "Your recent hydration is below a common daily intake target. "
            "You may want to drink water more regularly during the day."
        ),
        "MEDIUM": (
            "Your recent hydration is a bit below a common daily range. "
            "Consider keeping water nearby and sipping throughout the day."
        ),
        "LOW": (
            "Your hydration looks slightly under a common daily target. "
            "A few extra glasses of water may help."
        ),
    },
    "REST": {
        "HIGH": (
            "Your recent sleep duration is well below a common nightly target. "
            "Consider protecting a longer sleep window."
        ),
        "MEDIUM": (
            "Your recent sleep duration is below a common nightly target. "
            "You may want to keep a more consistent bedtime routine."
        ),
        "LOW": (
            "Your recent pattern shows room for better rest. "
            "Consider winding down a little earlier when possible."
        ),
    },
    "STRESS": {
        "HIGH": (
            "Your recent pattern shows elevated stress. "
            "You may want to schedule lighter tasks and short recovery breaks."
        ),
        "MEDIUM": (
            "Your recent stress ratings are on the higher side. "
            "Consider adding short recovery breaks between dense work blocks."
        ),
        "LOW": (
            "Your recent pattern shows mild stress elevation. "
            "A short pause or lighter block may help balance the day."
        ),
    },
    "ACTIVITY": {
        "HIGH": (
            "Your recent activity minutes are well below a common weekly movement target. "
            "You may want to add light movement you enjoy."
        ),
        "MEDIUM": (
            "Your recent activity minutes are below a common weekly movement target. "
            "Consider adding a few short walking or mobility sessions."
        ),
        "LOW": (
            "Your recent movement is a bit under a common activity range. "
            "Short walks can be a gentle way to stay active."
        ),
    },
    "MOOD": {
        "HIGH": (
            "Your recent mood ratings are quite low. "
            "You may want to keep lighter plans and include activities you usually enjoy."
        ),
        "MEDIUM": (
            "Your recent mood ratings are on the low side. "
            "You may want to keep lighter plans and include activities you usually enjoy."
        ),
        "LOW": (
            "Your recent mood looks slightly subdued. "
            "Consider making a little space for something restorative."
        ),
    },
}


def message_for(rec_type: str, priority: str) -> str:
    bank = MESSAGES.get(rec_type) or MESSAGES["REST"]
    return bank.get(priority) or bank["MEDIUM"]


def priority_from_score(score: float) -> str:
    if score >= PRIORITY_HIGH:
        return "HIGH"
    if score >= PRIORITY_MEDIUM:
        return "MEDIUM"
    return "LOW"
