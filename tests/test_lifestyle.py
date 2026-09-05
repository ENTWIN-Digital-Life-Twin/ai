from tests.regression_samples import LEGACY_LIFESTYLE_SAMPLES


def test_wellness_lifestyle_risk(client):
    payload = {
        "averageSleepMinutes": 420,
        "averageHydrationMl": 1700,
        "weeklyWorkoutMinutes": 180,
        "averageStress": 6.2,
        "averageFatigue": 5.4,
        "averageMood": 6.8,
        "averageDailySteps": 6200,
    }
    response = client.post("/api/v1/ai/lifestyle-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "RULE_BASED_BASELINE"
    assert body["riskLevel"] in {"LOW", "MODERATE", "HIGH"}
    assert body["score"] is not None
    assert body["modelVersion"] is None


def test_missing_optional_fields_are_not_treated_as_zero(client):
    response = client.post("/api/v1/ai/lifestyle-risk", json={"averageStress": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "RULE_BASED_BASELINE"
    assert body["riskLevel"] == "LOW"
    assert "LOW_SLEEP" not in body["factors"]
    assert "LOW_HYDRATION" not in body["factors"]
    assert "LOW_ACTIVITY" not in body["factors"]


def test_high_stress_only(client):
    response = client.post("/api/v1/ai/lifestyle-risk", json={"averageStress": 9})
    assert response.status_code == 200
    body = response.json()
    assert body["factors"] == ["HIGH_STRESS"]
    assert body["riskLevel"] == "MODERATE"


def test_empty_lifestyle_request_rejected(client):
    response = client.post("/api/v1/ai/lifestyle-risk", json={})
    assert response.status_code == 400
    assert response.json()["status"] == 400


def test_legacy_lifestyle_regression(client):
    for sample in LEGACY_LIFESTYLE_SAMPLES:
        response = client.post("/predict/lifestyle_risk", json=sample["request"])
        assert response.status_code == 200
        body = response.json()
        assert body["risk"] == sample["risk"]
        assert body["confidence"] == sample["confidence"]
        assert body["contributing_factors"] == sample["contributing_factors"]
        assert body["engine"] == "RULE_BASED_BASELINE"


def test_legacy_hyphen_path(client):
    response = client.post(
        "/predict/lifestyle-risk",
        json={"sleep_score": 85, "hydration_score": 80, "activity_score": 90, "stress_score": 88},
    )
    assert response.status_code == 200
    assert response.json()["risk"] == "normal"
