from __future__ import annotations

import re

from app.core.constants import ENGINE_RULE_BASED_BASELINE
from app.schemas.form_suggest import FormSuggestRequest, FormSuggestResponse, FormSuggestion

_WORK = ("report", "meeting", "email", "review", "deadline", "client", "sprint", "présentation", "rapport")
_STUDIES = ("exam", "chapter", "homework", "revision", "study", "td", "cours", "chapitre", "devoir")
_SPORT = ("run", "gym", "sport", "yoga", "workout", "jog", "footing", "musculation")
_PERSONAL = ("family", "home", "errand", "grocery", "famille", "maison")
_URGENT = ("urgent", "asap", "overdue", "important", "critique")


class FormSuggestService:
    def suggest(self, request: FormSuggestRequest) -> FormSuggestResponse:
        title = (request.title or "").strip()
        lowered = title.lower()
        category = (request.category or "").strip().upper()
        form_type = request.form_type
        items: list[FormSuggestion] = []

        if form_type == "TASK":
            items.extend(_task_suggestions(title, lowered, category, request.description))
        elif form_type == "EVENT":
            items.extend(_event_suggestions(title, lowered, category))
        elif form_type == "MEAL":
            items.extend(_meal_suggestions(title, lowered, category, request.description))
        elif form_type == "WORKOUT":
            items.extend(_workout_suggestions(title, lowered, category, request.description))
        elif form_type == "WELLNESS":
            items.extend(_wellness_suggestions(category or title))

        return FormSuggestResponse(suggestions=items[:6], engine=ENGINE_RULE_BASED_BASELINE)


def _task_suggestions(title: str, lowered: str, category: str, description: str | None) -> list[FormSuggestion]:
    items: list[FormSuggestion] = []
    inferred_category = _infer_task_category(lowered, category)
    if inferred_category and inferred_category.lower() != category.lower():
        items.append(
            FormSuggestion(
                field="category",
                value=inferred_category,
                label=f"Category: {inferred_category}",
                reason="Matched from the title",
            )
        )
    duration = _task_duration(lowered, inferred_category or category)
    items.append(
        FormSuggestion(
            field="duration",
            value=str(duration),
            label=f"{duration} min",
            reason="Typical duration for this kind of task",
        )
    )
    if any(token in lowered for token in _URGENT):
        items.append(
            FormSuggestion(
                field="priority",
                value="high",
                label="Priority: high",
                reason="The title sounds time-sensitive",
            )
        )
    if title and not (description or "").strip():
        items.append(
            FormSuggestion(
                field="description",
                value=_task_description(title, inferred_category or category),
                label="Suggested description",
                reason="Drafted from the title",
            )
        )
    return items


def _event_suggestions(title: str, lowered: str, category: str) -> list[FormSuggestion]:
    items: list[FormSuggestion] = []
    duration = 90 if any(token in lowered for token in ("exam", "class", "cours", "td")) else 45
    if "lunch" in lowered or "déjeuner" in lowered:
        duration = 60
    items.append(
        FormSuggestion(
            field="durationMinutes",
            value=str(duration),
            label=f"{duration} min",
            reason="Typical length for this event",
        )
    )
    if any(token in lowered for token in ("meet", "sync", "réunion", "reunion")):
        items.append(
            FormSuggestion(
                field="location",
                value="Meeting room",
                label="Location: meeting room",
                reason="Default for meetings",
            )
        )
        items.append(
            FormSuggestion(
                field="reminder",
                value="15",
                label="Reminder: 15 min",
                reason="Arrive prepared",
            )
        )
    inferred = _infer_task_category(lowered, category)
    if inferred:
        items.append(
            FormSuggestion(
                field="category",
                value=inferred,
                label=f"Category: {inferred}",
                reason="Matched from the title",
            )
        )
    if title:
        items.append(
            FormSuggestion(
                field="description",
                value=f"Event: {title}. Add agenda or participants if needed.",
                label="Suggested description",
                reason="Drafted from the title",
            )
        )
    return items


_FOOD_PER_100G: dict[str, tuple[float, float, float, float]] = {
    "rice": (130, 2.7, 28, 0.3),
    "riz": (130, 2.7, 28, 0.3),
    "chicken": (165, 31, 0, 3.6),
    "poulet": (165, 31, 0, 3.6),
    "pasta": (131, 5, 25, 1.1),
    "pates": (131, 5, 25, 1.1),
    "bread": (265, 9, 49, 3.2),
    "egg": (155, 13, 1.1, 11),
    "yogurt": (59, 10, 3.6, 0.4),
    "yaourt": (59, 10, 3.6, 0.4),
    "banana": (89, 1.1, 23, 0.3),
    "apple": (52, 0.3, 14, 0.2),
    "salmon": (208, 20, 0, 13),
    "saumon": (208, 20, 0, 13),
    "oats": (389, 17, 66, 7),
    "avoine": (389, 17, 66, 7),
}

_KCAL_PER_MIN = {
    "running": 11,
    "walking": 5,
    "cycling": 8,
    "gym": 7,
    "stretching": 3,
}


def _meal_suggestions(title: str, lowered: str, category: str, description: str | None) -> list[FormSuggestion]:
    blob = f"{title} {description or ''}".strip()
    parsed = _parse_food_quantities(blob)
    if parsed:
        calories = protein = carbs = fat = 0.0
        foods: list[str] = []
        for name, grams in parsed:
            per = _FOOD_PER_100G.get(name.lower())
            factor = grams / 100.0
            if per:
                calories += per[0] * factor
                protein += per[1] * factor
                carbs += per[2] * factor
                fat += per[3] * factor
            foods.append(f"{name.title()} {int(grams)}g")
        return [
            FormSuggestion(field="foods", value=", ".join(foods), label=", ".join(foods), reason="Parsed from quantities"),
            FormSuggestion(
                field="calories",
                value=str(int(round(calories))),
                label=f"{int(round(calories))} kcal",
                reason="Calculated from food quantities",
            ),
            FormSuggestion(
                field="protein",
                value=str(int(round(protein))),
                label=f"{int(round(protein))} g protein",
                reason="Calculated from food quantities",
            ),
            FormSuggestion(
                field="carbs",
                value=str(int(round(carbs))),
                label=f"{int(round(carbs))} g carbs",
                reason="Calculated from food quantities",
            ),
            FormSuggestion(
                field="fat",
                value=str(int(round(fat))),
                label=f"{int(round(fat))} g fat",
                reason="Calculated from food quantities",
            ),
        ]

    meal_type = category.lower() or _infer_meal_type(lowered)
    presets = {
        "breakfast": ("Oatmeal, fruit, yogurt", "420", "22", "55", "12"),
        "lunch": ("Grilled chicken, rice, vegetables", "650", "40", "70", "18"),
        "snack": ("Greek yogurt, almonds", "220", "14", "16", "10"),
        "dinner": ("Salmon, salad, potatoes", "580", "38", "45", "22"),
    }
    foods, calories, protein, carbs, fat = presets.get(meal_type, presets["lunch"])
    return [
        FormSuggestion(field="foods", value=foods, label=foods, reason="Balanced default"),
        FormSuggestion(field="calories", value=calories, label=f"{calories} kcal", reason="Typical for this meal"),
        FormSuggestion(field="protein", value=protein, label=f"{protein} g protein", reason="Typical for this meal"),
        FormSuggestion(field="carbs", value=carbs, label=f"{carbs} g carbs", reason="Typical for this meal"),
        FormSuggestion(field="fat", value=fat, label=f"{fat} g fat", reason="Typical for this meal"),
    ]


def _parse_food_quantities(text: str) -> list[tuple[str, float]]:
    matches = re.findall(r"([A-Za-zÀ-ÿ]+)\s+(\d+(?:\.\d+)?)\s*g\b", text, flags=re.IGNORECASE)
    lines: list[tuple[str, float]] = []
    for name, grams in matches:
        key = name.lower()
        if key in _FOOD_PER_100G:
            lines.append((name, float(grams)))
    return lines


def _workout_suggestions(title: str, lowered: str, category: str, description: str | None) -> list[FormSuggestion]:
    kind = category.lower() or _infer_workout(lowered)
    blob = f"{title} {description or ''}"
    duration_match = re.search(r"(\d+)\s*(?:min|minutes?)\b", blob, flags=re.IGNORECASE)
    duration = int(duration_match.group(1)) if duration_match else (45 if kind in {"gym", "cycling"} else 30)
    kcal_per_min = _KCAL_PER_MIN.get(kind, 6)
    calories = str(int(round(kcal_per_min * max(5, duration))))
    intensity = "high" if kind == "running" or duration >= 50 else "low" if kind in {"walking", "stretching"} else "medium"
    return [
        FormSuggestion(field="duration", value=str(duration), label=f"{duration} min", reason="From the session length"),
        FormSuggestion(field="calories", value=calories, label=f"{calories} kcal", reason="Calculated from type and duration"),
        FormSuggestion(field="intensity", value=intensity, label=f"Intensity: {intensity}", reason="Matched to the activity"),
    ]


def _wellness_suggestions(kind: str) -> list[FormSuggestion]:
    lowered = kind.lower()
    if "hydrat" in lowered or "water" in lowered:
        return [FormSuggestion(field="ml", value="250", label="250 ml", reason="A standard glass")]
    if "sleep" in lowered:
        return [
            FormSuggestion(field="bed", value="23:00", label="Bedtime 23:00", reason="Aligned with the default target"),
            FormSuggestion(field="wake", value="07:00", label="Wake 07:00", reason="About 8 hours of sleep"),
        ]
    if "activ" in lowered:
        return [FormSuggestion(field="minutes", value="30", label="30 min", reason="Daily activity target")]
    if "mood" in lowered:
        return [FormSuggestion(field="mood", value="4", label="Mood 4/5", reason="Neutral-positive check-in")]
    if "stress" in lowered:
        return [FormSuggestion(field="stress", value="low", label="Stress: low", reason="Default check-in")]
    return []


def _infer_task_category(lowered: str, current: str) -> str | None:
    if any(token in lowered for token in _SPORT):
        return "sport"
    if any(token in lowered for token in _STUDIES):
        return "studies"
    if any(token in lowered for token in _PERSONAL):
        return "personal"
    if any(token in lowered for token in _WORK):
        return "work"
    if current:
        return None
    return "work" if lowered else None


def _task_duration(lowered: str, category: str) -> int:
    if any(token in lowered for token in ("report", "rapport", "exam", "présentation")):
        return 90
    if any(token in lowered for token in ("email", "mail", "call")):
        return 20
    if "sport" in category.lower() or any(token in lowered for token in _SPORT):
        return 45
    if "studies" in category.lower():
        return 60
    return 45


def _task_description(title: str, category: str) -> str:
    return f"Complete “{title}” ({category or 'task'}). Break it into small steps and set a realistic end time."


def _infer_meal_type(lowered: str) -> str:
    if any(token in lowered for token in ("breakfast", "petit-déj", "matin")):
        return "breakfast"
    if any(token in lowered for token in ("dinner", "dîner", "diner", "soir")):
        return "dinner"
    if any(token in lowered for token in ("snack", "collation")):
        return "snack"
    return "lunch"


def _infer_workout(lowered: str) -> str:
    if "walk" in lowered or "marche" in lowered:
        return "walking"
    if "cycl" in lowered or "vélo" in lowered or "velo" in lowered:
        return "cycling"
    if "gym" in lowered or "muscul" in lowered:
        return "gym"
    if "yoga" in lowered or "stretch" in lowered:
        return "stretching"
    return "running"
