import os

import pytest

pytestmark = pytest.mark.live

LIVE_BASE = os.getenv("AI_LIVE_URL", "http://127.0.0.1:8090")


@pytest.fixture
def live_client():
    if os.getenv("AI_LIVE") != "1":
        pytest.skip("Set AI_LIVE=1 and start the AI service to run live smoke tests.")
    import httpx

    with httpx.Client(base_url=LIVE_BASE, timeout=10.0) as client:
        health = client.get("/health")
        if health.status_code != 200:
            pytest.skip(f"AI service is not healthy at {LIVE_BASE}")
        yield client


def test_live_health(live_client):
    body = live_client.get("/health").json()
    assert body["status"] == "UP"


def test_live_lifestyle_from_java_shape(live_client):
    response = live_client.post(
        "/api/v1/ai/lifestyle-risk",
        json={"averageSleepMinutes": 420, "averageHydrationMl": 1700, "averageStress": 6.2},
    )
    assert response.status_code == 200
    assert response.json()["engine"] == "RULE_BASED_BASELINE"


def test_live_recommendations_from_java_shape(live_client):
    response = live_client.post("/api/v1/ai/recommendations", json={"hydrationMl": 900})
    assert response.status_code == 200
    assert response.json()["engine"] == "RULE_BASED_BASELINE"


def test_live_task_duration_from_java_shape(live_client):
    response = live_client.post(
        "/api/v1/ai/task-duration",
        json={"complexity": "HARD", "energyRequired": "HIGH", "userEstimateMinutes": 60},
    )
    assert response.status_code == 200
    assert response.json()["engine"] == "BASELINE_ESTIMATOR"


def test_live_chat_requires_ollama(live_client):
    response = live_client.post("/api/v1/ai/chat", json={"question": "Give me one planning tip."})
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert response.json()["answer"]
        assert "disclaimer" in response.json()
