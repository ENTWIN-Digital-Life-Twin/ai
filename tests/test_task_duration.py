def test_valid_baseline_prediction(client):
    payload = {
        "category": "STUDY",
        "complexity": "HARD",
        "energyRequired": "HIGH",
        "userEstimateMinutes": 90,
        "historicalAverageMinutes": 110,
    }
    response = client.post("/api/v1/ai/task-duration", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "BASELINE_ESTIMATOR"
    assert body["confidence"] is None
    assert 90 <= body["predictedDurationMinutes"] <= 140


def test_negative_duration_rejected(client):
    response = client.post(
        "/api/v1/ai/task-duration",
        json={"userEstimateMinutes": -10},
    )
    assert response.status_code == 422


def test_deterministic_without_estimates(client):
    payload = {"category": "STUDY", "complexity": "HARD", "energyRequired": "HIGH"}
    first = client.post("/api/v1/ai/task-duration", json=payload)
    second = client.post("/api/v1/ai/task-duration", json=payload)
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["engine"] == "BASELINE_ESTIMATOR"
    assert first.json()["predictedDurationMinutes"] == 135


def test_priority_is_ignored_if_sent(client):
    payload = {"category": "WORK", "complexity": "MEDIUM", "priority": "HIGH"}
    response = client.post("/api/v1/ai/task-duration", json=payload)
    assert response.status_code == 200
    assert response.json()["predictedDurationMinutes"] == 60
