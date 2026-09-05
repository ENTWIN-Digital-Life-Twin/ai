"""Rule-based lifestyle baseline used by the original teammate implementation.

Production lifestyle-risk still uses this engine (RULE_BASED_BASELINE).
Do not treat this as a trained ML model.
"""


def evaluate_risk(sleep_score, hydration_score, activity_score, stress_score, threshold=60):
    scores = {
        "fatigue": sleep_score,
        "deshydratation": hydration_score,
        "sedentarite": activity_score,
        "surcharge": stress_score,
    }

    below_threshold = {domain: val for domain, val in scores.items() if val < threshold}

    if len(below_threshold) == 0:
        avg_score = sum(scores.values()) / len(scores)
        return {"risk": "normal", "confidence": round(avg_score / 100, 2), "contributing_factors": []}

    if len(below_threshold) >= 2:
        avg_gap = sum(threshold - val for val in below_threshold.values()) / len(below_threshold)
        confidence = round(min(avg_gap / threshold, 1.0), 2)
        return {"risk": "surcharge", "confidence": confidence, "contributing_factors": list(below_threshold.keys())}

    lowest_domain = list(below_threshold.keys())[0]
    lowest_value = below_threshold[lowest_domain]
    confidence = round((threshold - lowest_value) / threshold, 2)
    return {"risk": lowest_domain, "confidence": confidence, "contributing_factors": [lowest_domain]}


if __name__ == "__main__":
    cas_de_test = [
        {"sleep_score": 45, "hydration_score": 70, "activity_score": 80, "stress_score": 65},  # 1 seul bas -> fatigue
        {"sleep_score": 50, "hydration_score": 45, "activity_score": 60, "stress_score": 55},  # plusieurs bas -> surcharge
        {"sleep_score": 85, "hydration_score": 80, "activity_score": 90, "stress_score": 88},  # normal
        {"sleep_score": 90, "hydration_score": 85, "activity_score": 20, "stress_score": 75},  # 1 seul bas -> sedentarite
        {"sleep_score": 40, "hydration_score": 35, "activity_score": 30, "stress_score": 25},  # tous bas -> surcharge
    ]

    for cas in cas_de_test:
        resultat = evaluate_risk(**cas)
        print(cas, "->", resultat)