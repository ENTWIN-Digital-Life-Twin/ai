import joblib

from tests.conftest import SLEEP_SAMPLE, SLEEP_SAMPLE_SNAKE
from tests.regression_samples import SLEEP_REGRESSION_SAMPLES


def test_valid_sleep_risk(client):
    response = client.post("/api/v1/ai/sleep-risk", json=SLEEP_SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["predictedClass"] == "None"
    assert body["riskLevel"] == "LOW"
    assert body["modelProbability"] == 0.82
    assert body["modelName"] == "sleep-risk"
    assert body["modelVersion"] == "1.0.0"
    assert "not a medical diagnosis" in body["disclaimer"]
    assert "diagnosed" not in body["disclaimer"].lower()


def test_sleep_accepts_snake_case(client):
    response = client.post("/api/v1/ai/sleep-risk", json=SLEEP_SAMPLE_SNAKE)
    assert response.status_code == 200
    assert response.json()["predictedClass"] == "None"


def test_invalid_age_rejected(client):
    payload = dict(SLEEP_SAMPLE)
    payload["age"] = 0
    response = client.post("/api/v1/ai/sleep-risk", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == 422
    assert "traceback" not in body.get("message", "").lower()
    assert "fieldErrors" in body


def test_invalid_gender_rejected(client):
    payload = dict(SLEEP_SAMPLE)
    payload["gender"] = "Unknown"
    response = client.post("/api/v1/ai/sleep-risk", json=payload)
    assert response.status_code == 422


def test_normal_weight_bmi_alias(client):
    payload = dict(SLEEP_SAMPLE)
    payload["bmiCategory"] = "Normal Weight"
    response = client.post("/api/v1/ai/sleep-risk", json=payload)
    assert response.status_code == 200


def test_regression_samples_unchanged(client):
    for sample in SLEEP_REGRESSION_SAMPLES:
        response = client.post("/api/v1/ai/sleep-risk", json=sample["request"])
        assert response.status_code == 200, sample["name"]
        body = response.json()
        assert body["predictedClass"] == sample["predicted_class"], sample["name"]
        assert body["modelProbability"] == sample["confidence"], sample["name"]


def test_legacy_sleep_endpoints(client):
    for path in ("/predict/sleep_disorder", "/predict/sleep-disorder"):
        response = client.post(path, json=SLEEP_SAMPLE_SNAKE)
        assert response.status_code == 200
        body = response.json()
        assert body["risk"] == "None"
        assert body["confidence"] == 0.82
        assert "disclaimer" in body


def test_model_cached_across_requests(client):
    first = id(client.app.state.models.sleep.estimator)
    client.post("/api/v1/ai/sleep-risk", json=SLEEP_SAMPLE)
    client.post("/api/v1/ai/sleep-risk", json=SLEEP_SAMPLE)
    assert id(client.app.state.models.sleep.estimator) == first
    assert client.app.state.models.sleep.feature_names is not None


def test_sleep_features_match_joblib(client):
    loaded = list(joblib.load("app/models/sleep_disorder_features.joblib"))
    assert client.app.state.models.sleep.feature_names == loaded
    assert client.app.state.models.sleep.estimator.n_features_in_ == len(loaded)
