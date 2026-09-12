def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UP"
    assert body["service"] == "entwin-ai-service"
    assert body["version"] == "0.1.0"


def test_versioned_health(client):
    response = client.get("/api/v1/ai/health")
    assert response.status_code == 200
    assert response.json()["service"] == "entwin-ai-service"


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "AI service is running"


def test_models_catalog(client):
    response = client.get("/api/v1/ai/models")
    assert response.status_code == 200
    models = {item["name"]: item for item in response.json()["models"]}
    assert models["sleep-risk"]["loaded"] is True
    assert models["sleep-risk"]["algorithm"] == "DecisionTreeClassifier"
    assert models["sleep-risk"]["featureCount"] == 11
    assert models["sleep-risk"]["usedInProduction"] is True
    assert models["lifestyle-risk"]["usedInProduction"] is False
    assert models["assistant"]["algorithm"] == "qwen2.5:7b"
    assert models["assistant"]["usedInProduction"] is True
    payload = response.text
    assert "joblib" not in payload
    assert "C:\\" not in payload
    assert "/app/models" not in payload
