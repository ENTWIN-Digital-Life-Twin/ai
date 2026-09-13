import pytest


@pytest.mark.parametrize(
    "payload,expected_engine,expected_risk",
    [
        (
            {
                "averageSleepMinutes": 420,
                "averageHydrationMl": 1700,
                "weeklyWorkoutMinutes": 180,
                "averageStress": 6.2,
                "averageFatigue": 5.4,
                "averageMood": 6.8,
                "averageDailySteps": 6200,
            },
            "RULE_BASED_BASELINE",
            {"LOW", "MODERATE", "HIGH"},
        ),
        ({"averageStress": 3}, "RULE_BASED_BASELINE", {"LOW"}),
        ({"averageStress": 9}, "RULE_BASED_BASELINE", {"MODERATE"}),
            ({"averageSleepMinutes": 200, "averageHydrationMl": 800, "weeklyWorkoutMinutes": 10}, "RULE_BASED_BASELINE", {"HIGH"}),
    ],
)
def test_java_lifestyle_payloads(client, payload, expected_engine, expected_risk):
    response = client.post("/api/v1/ai/lifestyle-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == expected_engine
    assert body["riskLevel"] in expected_risk
    assert "userId" not in body
    assert "traceback" not in str(body).lower()


@pytest.mark.parametrize(
    "payload,expected_types",
    [
        ({"hydrationMl": 900}, {"HYDRATION"}),
        ({"sleepMinutes": 330}, {"REST"}),
        ({"stressLevel": 8}, {"STRESS", "REST"}),
        ({"weeklyWorkoutMinutes": 10}, {"ACTIVITY"}),
        ({"moodLevel": 3}, {"MOOD"}),
        ({"dailySteps": 2000}, {"ACTIVITY"}),
        (
            {
                "sleepMinutes": 480,
                "hydrationMl": 2100,
                "stressLevel": 3,
                "fatigueLevel": 3,
                "weeklyWorkoutMinutes": 180,
                "moodLevel": 8,
                "dailySteps": 9000,
            },
            set(),
        ),
    ],
)
def test_java_recommendation_payloads(client, payload, expected_types):
    response = client.post("/api/v1/ai/recommendations", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "RULE_BASED_BASELINE"
    types = {item["type"] for item in body["recommendations"]}
    if expected_types:
        assert expected_types & types
    else:
        assert types == set()
    joined = " ".join(item["message"].lower() for item in body["recommendations"])
    assert "diagnos" not in joined
    assert "prescription" not in joined


@pytest.mark.parametrize(
    "payload",
    [
        {"complexity": "HARD", "energyRequired": "HIGH", "userEstimateMinutes": 60},
        {"complexity": "EASY", "energyRequired": "LOW", "userEstimateMinutes": 30},
        {"complexity": "MEDIUM", "userEstimateMinutes": 45, "historicalAverageMinutes": 50},
        {"category": "STUDY", "complexity": "HARD", "energyRequired": "HIGH"},
    ],
)
def test_java_task_duration_payloads(client, payload):
    response = client.post("/api/v1/ai/task-duration", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "BASELINE_ESTIMATOR"
    assert isinstance(body["predictedDurationMinutes"], int)
    assert body["predictedDurationMinutes"] > 0
    assert "userId" not in body


def test_java_client_empty_lifestyle_is_rejected(client):
    response = client.post("/api/v1/ai/lifestyle-risk", json={})
    assert response.status_code == 400


def test_chat_accepts_frontend_shaped_context(client, fake_llm):
    payload = {
        "question": "Should I keep tomorrow morning light?",
        "context": {
            "sleepRisk": {"predictedClass": "None", "riskLevel": "LOW"},
            "lifestyleRisk": {"riskLevel": "MODERATE", "factors": ["HIGH_STRESS"]},
            "recommendations": [{"type": "STRESS", "priority": "HIGH", "message": "Schedule lighter tasks."}],
            "planning": {"tasksTotal": 3, "productivityPercent": 50},
        },
    }
    response = client.post("/api/v1/ai/chat", json=payload)
    assert response.status_code == 200
    user_content = fake_llm.last_messages[1].content
    assert "HIGH_STRESS" in user_content
    assert "Schedule lighter tasks" in user_content
    assert "productivityPercent" in user_content
