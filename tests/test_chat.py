from unittest.mock import patch

import httpx
import pytest

from app.core.exceptions import LLMUnavailableError
from app.llm.ollama_provider import OllamaProvider
from app.llm.provider import ChatMessage


def test_chat_returns_answer(client, fake_llm):
    response = client.post("/api/v1/ai/chat", json={"question": "How do I create a task?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ok"
    assert body["provider"] == "ollama"
    assert body["model"] == "qwen2.5:7b"
    assert body["engine"] == "LLM_PROVIDER"
    assert "not a medical diagnosis" in body["disclaimer"].lower()
    assert fake_llm.calls == 1


def test_chat_sends_only_authorized_context(client, fake_llm):
    payload = {
        "question": "What is my sleep risk?",
        "context": {"sleepRisk": {"riskLevel": "LOW", "predictedClass": "None"}},
    }
    response = client.post("/api/v1/ai/chat", json=payload)
    assert response.status_code == 200
    system_content = fake_llm.last_messages[0].content
    user_content = fake_llm.last_messages[1].content
    assert "Use ONLY the authorized context" in system_content
    assert "never diagnose" in system_content.lower()
    assert "LOW" in user_content
    assert "None" in user_content
    assert "userId" not in user_content


def test_legacy_assistant_alias(client, fake_llm):
    response = client.post("/api/assistant/chat", json={"question": "Hello"})
    assert response.status_code == 200
    assert response.json()["answer"] == "ok"


def test_blank_question_is_rejected(client, fake_llm):
    response = client.post("/api/v1/ai/chat", json={"question": "   "})
    assert response.status_code == 422
    assert fake_llm.calls == 0


def test_emergency_uses_rules_not_llm(client, fake_llm):
    response = client.post("/api/v1/ai/chat", json={"question": "I have chest pain right now"})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "RULE_BASED_BASELINE"
    assert body["proposedAction"] == "CONTACT_EMERGENCY_SERVICES"
    assert "emergency" in body["answer"].lower()
    assert fake_llm.calls == 0


def test_chat_unavailable_when_provider_missing(client):
    original = client.app.state.llm
    client.app.state.llm = None
    try:
        response = client.post("/api/v1/ai/chat", json={"question": "Plan my morning"})
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "RULE_BASED_BASELINE"
        assert body["answer"]
        assert "snapshot" not in body["answer"].lower()
        assert "task" in body["answer"].lower() or "morning" in body["answer"].lower()
    finally:
        client.app.state.llm = original


def test_organize_afternoon_uses_tasks_not_snapshot_dump(client):
    original = client.app.state.llm
    client.app.state.llm = None
    try:
        response = client.post(
            "/api/v1/ai/chat",
            json={
                "question": "How should I organize my afternoon?",
                "context": {
                    "planning": {
                        "tasksCompleted": 2,
                        "tasksTotal": 7,
                        "freeMinutes": 90,
                        "overloaded": False,
                    },
                    "upcoming": {"title": "Team review", "time": "16:00"},
                },
            },
        )
        assert response.status_code == 200
        answer = response.json()["answer"].lower()
        assert "snapshot" not in answer
        assert "balance" not in answer
        assert "5 remaining" in answer or "remaining" in answer
        assert "team review" in answer
        assert "16:00" in answer
    finally:
        client.app.state.llm = original


def test_chat_rule_based_mentions_sleep_duration(client):
    original = client.app.state.llm
    client.app.state.llm = None
    try:
        response = client.post(
            "/api/v1/ai/chat",
            json={
                "question": "How is my sleep this week?",
                "context": {
                    "wellness": {"sleep": {"value": "7h 20"}},
                    "weeklySummary": {"averageSleepMinutes": 450},
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "RULE_BASED_BASELINE"
        assert "7h 20" in body["answer"]
        assert "duration" in body["answer"].lower()
        assert "7.5h" in body["answer"]
    finally:
        client.app.state.llm = original


def test_chat_falls_back_when_ollama_unavailable(client):
    original = client.app.state.llm

    class BoomLLM:
        def generate(self, messages):
            raise LLMUnavailableError("Ollama server is unreachable.")

    client.app.state.llm = BoomLLM()
    try:
        response = client.post("/api/v1/ai/chat", json={"question": "How is my sleep?"})
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "RULE_BASED_BASELINE"
        assert "bedtime" in body["answer"].lower() or "sleep" in body["answer"].lower()
    finally:
        client.app.state.llm = original


def test_create_task_intent_works_without_llm(client):
    original = client.app.state.llm
    client.app.state.llm = None
    try:
        response = client.post(
            "/api/v1/ai/chat",
            json={"question": "Create a task: finish the weekly report"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["proposedAction"] == "CREATE_TASK"
        assert body["proposedTask"]["title"].lower().find("weekly report") >= 0
        assert body["engine"] == "RULE_BASED_BASELINE"
    finally:
        client.app.state.llm = original


def test_chat_forwards_history_to_llm(client, fake_llm):
    response = client.post(
        "/api/v1/ai/chat",
        json={
            "question": "And today?",
            "history": [
                {"role": "user", "content": "How was my sleep yesterday?"},
                {"role": "assistant", "content": "You slept about 7 hours."},
            ],
            "context": {"weeklySummary": {"averageSleepMinutes": 420}},
        },
    )
    assert response.status_code == 200
    roles = [item.role for item in fake_llm.last_messages]
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles
    assert any("yesterday" in item.content for item in fake_llm.last_messages)


def test_how_to_create_task_without_llm(client):
    original = client.app.state.llm
    client.app.state.llm = None
    try:
        response = client.post("/api/v1/ai/chat", json={"question": "How do I create a task?"})
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "RULE_BASED_BASELINE"
        assert body["proposedAction"] is None
        assert "Tasks" in body["answer"] or "Planning" in body["answer"]
    finally:
        client.app.state.llm = original


def test_how_to_create_task_does_not_propose_task(client, fake_llm):
    response = client.post("/api/v1/ai/chat", json={"question": "How do I create a task?"})
    assert response.status_code == 200
    body = response.json()
    assert body["proposedAction"] is None
    assert body["proposedTask"] is None
    assert fake_llm.calls == 1


def test_llm_entwin_action_is_parsed(client, fake_llm):
    fake_llm.text = (
        'Sure, I can add that.\n'
        'ENTWIN_ACTION:{"type":"CREATE_TASK","title":"Review chapter 5","durationMinutes":60,"priority":"HIGH","category":"STUDIES"}'
    )
    response = client.post("/api/v1/ai/chat", json={"question": "Please add a revision task"})
    assert response.status_code == 200
    body = response.json()
    assert "ENTWIN_ACTION" not in body["answer"]
    assert body["proposedAction"] == "CREATE_TASK"
    assert body["proposedTask"]["title"] == "Review chapter 5"
    assert body["proposedTask"]["durationMinutes"] == 60
    assert body["proposedTask"]["category"] == "STUDIES"


def test_form_suggest_meal_from_quantity(client):
    response = client.post(
        "/api/v1/ai/form-suggest",
        json={"formType": "MEAL", "title": "rice 100g chicken 150g", "category": "lunch"},
    )
    assert response.status_code == 200
    fields = {item["field"]: item["value"] for item in response.json()["suggestions"]}
    assert "Rice 100g" in fields["foods"]
    assert "Chicken 150g" in fields["foods"]
    assert int(fields["calories"]) > 0
    assert int(fields["protein"]) > 0


def test_form_suggest_workout_from_duration(client):
    response = client.post(
        "/api/v1/ai/form-suggest",
        json={"formType": "WORKOUT", "title": "running 40 min", "category": "running"},
    )
    assert response.status_code == 200
    fields = {item["field"]: item["value"] for item in response.json()["suggestions"]}
    assert fields["duration"] == "40"
    assert fields["calories"] == "440"
    assert fields["intensity"] == "high"


def test_form_suggest_task_from_title(client):
    response = client.post(
        "/api/v1/ai/form-suggest",
        json={"formType": "TASK", "title": "Urgent weekly report", "category": "personal"},
    )
    assert response.status_code == 200
    body = response.json()
    fields = {item["field"]: item["value"] for item in body["suggestions"]}
    assert fields["duration"] == "90"
    assert fields["priority"] == "high"
    assert fields["category"] == "work"
    assert body["engine"] == "RULE_BASED_BASELINE"


def test_models_catalog_includes_assistant(client):
    response = client.get("/api/v1/ai/models")
    assert response.status_code == 200
    models = {item["name"]: item for item in response.json()["models"]}
    assert models["assistant"]["usedInProduction"] is True
    assert models["assistant"]["algorithm"] == "qwen2.5:7b"
    assert "joblib" not in response.text


def test_ollama_provider_posts_to_remote_chat_api():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = request.read()
        return httpx.Response(
            200,
            json={"model": "qwen2.5:7b", "message": {"role": "assistant", "content": "Hello from Ollama"}},
        )

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5:7b",
        timeout_seconds=5,
        connect_timeout_seconds=1,
        temperature=0.2,
        max_tokens=64,
    )
    real_client = httpx.Client

    def client_factory(**kwargs):
        kwargs["transport"] = transport
        return real_client(**kwargs)

    with patch("app.llm.ollama_provider.httpx.Client", client_factory):
        result = provider.generate([ChatMessage(role="user", content="Hello")])
    assert result.text == "Hello from Ollama"
    assert result.provider == "ollama"
    assert "http://localhost:11434/api/chat" in captured["url"]


def test_ollama_provider_maps_connection_errors():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5:7b",
        timeout_seconds=5,
        connect_timeout_seconds=1,
        temperature=0.2,
        max_tokens=64,
    )

    class BoomClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            raise httpx.ConnectError("refused")

    with patch("app.llm.ollama_provider.httpx.Client", BoomClient):
        with pytest.raises(LLMUnavailableError):
            provider.generate([ChatMessage(role="user", content="Hello")])
