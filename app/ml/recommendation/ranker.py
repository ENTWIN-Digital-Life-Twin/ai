"""Rank recommendation types from model probabilities."""

from __future__ import annotations

from dataclasses import dataclass

from app.ml.recommendation.catalog import (
    REC_TYPES,
    SCORE_FLOOR,
    SEVERITY_WEIGHT,
    TOP_K,
    message_for,
    priority_from_score,
)


@dataclass(frozen=True)
class RankedRecommendation:
    type: str
    priority: str
    message: str
    score: float
    confidence: float


def rank_recommendations(
    probabilities: dict[str, float],
    *,
    top_k: int = TOP_K,
    score_floor: float = SCORE_FLOOR,
) -> list[RankedRecommendation]:
    """Rank by P(type) * severity; keep at most one item per type; cap at top_k."""
    ranked: list[tuple[str, float]] = []
    for rec_type in REC_TYPES:
        prob = float(probabilities.get(rec_type, 0.0))
        weight = SEVERITY_WEIGHT.get(rec_type, 1.0)
        score = min(1.0, max(0.0, prob) * weight)
        if score >= score_floor:
            ranked.append((rec_type, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    results: list[RankedRecommendation] = []
    seen: set[str] = set()
    for rec_type, score in ranked:
        if rec_type in seen:
            continue
        seen.add(rec_type)
        priority = priority_from_score(score)
        results.append(
            RankedRecommendation(
                type=rec_type,
                priority=priority,
                message=message_for(rec_type, priority),
                score=round(score, 4),
                confidence=round(min(1.0, score), 4),
            )
        )
        if len(results) >= top_k:
            break
    return results
