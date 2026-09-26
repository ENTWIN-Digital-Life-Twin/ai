def test_low_hydration_recommendation(client):
    response = client.post("/api/v1/ai/recommendations", json={"hydrationMl": 900})
    assert response.status_code == 200
    body = response.json()
    types = [item["type"] for item in body["recommendations"]]
    assert "HYDRATION" in types
    assert body["engine"] in {"ML_RECOMMENDER", "RULE_BASED_BASELINE"}


def test_high_stress_recovery_recommendation(client):
    response = client.post("/api/v1/ai/recommendations", json={"stressLevel": 8})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert items
    joined = " ".join(item["message"].lower() for item in items)
    assert any(item["type"] in {"STRESS", "REST"} for item in items)
    assert "recovery" in joined or "rest" in joined or "lighter" in joined
    assert "diagnos" not in joined
    assert "you have" not in joined
    assert "you must" not in joined


def test_healthy_values_are_not_alarming(client):
    payload = {
        "sleepMinutes": 480,
        "hydrationMl": 2100,
        "stressLevel": 3,
        "fatigueLevel": 3,
        "weeklyWorkoutMinutes": 180,
        "moodLevel": 8,
        "dailySteps": 9000,
    }
    response = client.post("/api/v1/ai/recommendations", json=payload)
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert items == []
    blob = str(items).lower()
    assert "disease" not in blob
    assert "diagnos" not in blob


def test_short_sleep_produces_rest_recommendation(client):
    response = client.post("/api/v1/ai/recommendations", json={"sleepMinutes": 330})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert any(item["type"] == "REST" and item["priority"] in {"HIGH", "MEDIUM"} for item in items)


def test_ml_engine_when_model_loaded(client):
    response = client.post("/api/v1/ai/recommendations", json={"stressLevel": 9, "sleepMinutes": 300})
    assert response.status_code == 200
    body = response.json()
    # Model artifact is present in CI/dev after training; otherwise rules still work.
    assert body["engine"] in {"ML_RECOMMENDER", "RULE_BASED_BASELINE"}
    types = {item["type"] for item in body["recommendations"]}
    assert types & {"STRESS", "REST"}
    if body["engine"] == "ML_RECOMMENDER":
        for item in body["recommendations"]:
            assert "score" in item
            assert item["score"] is None or 0.0 <= item["score"] <= 1.0


def test_recommendation_messages_are_non_medical(client):
    response = client.post(
        "/api/v1/ai/recommendations",
        json={"hydrationMl": 800, "moodLevel": 2, "weeklyWorkoutMinutes": 5},
    )
    assert response.status_code == 200
    joined = " ".join(item["message"].lower() for item in response.json()["recommendations"])
    assert "diagnos" not in joined
    assert "prescription" not in joined
    assert "disease" not in joined
