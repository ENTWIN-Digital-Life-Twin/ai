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
        assert response.status_code == 503
        assert "unavailable" in response.json()["error"].lower() or "unavailable" in response.json()["message"].lower()
    finally:
        client.app.state.llm = original


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
