from fastapi.testclient import TestClient
import pytest

from app.main import app

SLEEP_SAMPLE = {
    "age": 24,
    "gender": "Female",
    "sleepDuration": 7.2,
    "qualityOfSleep": 8,
    "physicalActivityLevel": 60,
    "stressLevel": 4,
    "bmiCategory": "Normal",
    "heartRate": 72,
    "dailySteps": 8000,
}

SLEEP_SAMPLE_SNAKE = {
    "age": 24,
    "gender": "Female",
    "sleep_duration": 7.2,
    "quality_of_sleep": 8,
    "physical_activity_level": 60,
    "stress_level": 4,
    "bmi_category": "Normal",
    "heart_rate": 72,
    "daily_steps": 8000,
}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
