from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import llm_client as llm_api
from axiom.api import openai_compat
from axiom.core import llm_client


def _client(*routers) -> TestClient:
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    return TestClient(app)


def test_provider_catalog_endpoint_is_read_only_and_secret_free() -> None:
    response = _client(llm_api.router).get("/api/llm/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["credentials_persisted"] is False
    assert {item["id"] for item in body["providers"]} >= {
        "axiom-local", "openai", "anthropic", "gemini", "deepseek", "moonshot", "openai-compatible"
    }


def test_cloud_chat_endpoint_rejects_missing_consent_before_invoking_client(monkeypatch) -> None:
    def unexpected(*_args, **_kwargs):
        raise AssertionError("cloud inference must not run without consent")

    monkeypatch.setattr(llm_client, "chat", unexpected)
    response = _client(llm_api.router).post("/api/llm/chat", json={
        "provider": "deepseek",
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "private question"}],
        "api_key": "test-secret",
        "consent": False,
    })

    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()
    assert "test-secret" not in response.text


def test_local_model_inventory_endpoint_returns_client_result(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "list_models", lambda provider, api_key=None, base_url=None: {
        "provider": provider, "models": ["tiny"], "credential_storage": "not_applicable", "execution": "local"
    })
    response = _client(llm_api.router).post("/api/llm/models", json={"provider": "axiom-local"})

    assert response.status_code == 200
    assert response.json()["models"] == ["tiny"]


def test_llm_chat_endpoint_returns_normalized_result(monkeypatch) -> None:
    captured = {}

    def fake_chat(provider, model, messages, **kwargs):
        captured.update({"provider": provider, "model": model, "messages": messages, **kwargs})
        return {
            "provider": "axiom-local",
            "model": "tiny",
            "response": "hello",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "credential_storage": "not_applicable",
            "execution": "local",
            "device": "cpu",
            "generated_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    response = _client(llm_api.router).post("/api/llm/chat", json={
        "provider": "axiom-local",
        "model": "tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0,
        "top_p": 0.82,
        "top_k": 32,
        "min_p": 0.23,
        "repetition_penalty": 1.15,
        "seed": 13,
        "stop_sequences": ["END"],
    })

    assert response.status_code == 200
    assert response.json()["response"] == "hello"
    assert captured["device"] == "auto"
    assert captured["top_p"] == 0.82
    assert captured["top_k"] == 32
    assert captured["min_p"] == 0.23
    assert captured["repetition_penalty"] == 1.15
    assert captured["seed"] == 13
    assert captured["stop_sequences"] == ["END"]


def test_ollama_local_chat_does_not_require_cloud_consent(monkeypatch) -> None:
    captured = {}

    def fake_chat(provider, model, messages, **kwargs):
        captured.update({"provider": provider, "model": model, "messages": messages, **kwargs})
        return {
            "provider": "ollama-local",
            "model": model,
            "response": "local Ollama response",
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "credential_storage": "not_applicable",
            "execution": "local",
            "device": "ollama",
            "generated_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    response = _client(llm_api.router).post("/api/llm/chat", json={
        "provider": "ollama-local",
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": "hello local model"}],
        "consent": False,
        "max_tokens": 16,
        "temperature": 0,
    })

    assert response.status_code == 200
    assert response.json()["execution"] == "local"
    assert captured["provider"] == "ollama-local"
    assert captured["consent"] is False


def test_openai_compatible_bridge_requires_bearer_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_OPENAI_API_TOKEN", "bridge-token-sentinel")
    monkeypatch.setattr(openai_compat, "_native_models", lambda: [{
        "name": "tiny", "path": "/trusted/tiny", "model_type": "llama"
    }])
    client = _client(openai_compat.router)

    unauthorized = client.get("/v1/models")
    bad_token = client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"})
    authorized = client.get("/v1/models", headers={"Authorization": "Bearer bridge-token-sentinel"})

    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert bad_token.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["data"][0]["id"] == "tiny"
    assert "bridge-token-sentinel" not in authorized.text


def test_llm_chat_api_rejects_invalid_sampling_limits() -> None:
    for body in (
        {"top_p": 0},
        {"top_k": -1},
        {"min_p": -0.01},
        {"min_p": 1.01},
        {"repetition_penalty": 0},
        {"seed": -1},
        {"stop_sequences": ["x" * 257]},
        {"stop_sequences": ["STOP"] * 17},
    ):
        response = _client(llm_api.router).post("/api/llm/chat", json={
            "provider": "axiom-local",
            "model": "tiny",
            "messages": [{"role": "user", "content": "test"}],
            **body,
        })
        assert response.status_code == 422, body


def test_openai_native_n_choices_use_one_batched_run_with_full_chat_history(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(openai_compat, "_native_models", lambda: [{
        "name": "tiny", "path": "/trusted/tiny", "model_type": "llama",
    }])

    def fake_batch(model_path, prompts, **kwargs):
        captured.update(model_path=model_path, prompts=prompts, **kwargs)
        return {
            "model": "tiny",
            "batch_size": 3,
            "prompt_tokens": 15,
            "completion_tokens": 9,
            "results": [
                {"index": 1, "response": "choice A", "prompt_tokens": 5, "completion_tokens": 2, "stop_reason": "max_new_tokens"},
                {"index": 2, "response": "choice B", "prompt_tokens": 5, "completion_tokens": 3, "stop_reason": "eos_token"},
                {"index": 3, "response": "choice C", "prompt_tokens": 5, "completion_tokens": 4, "stop_reason": "stop_sequence"},
            ],
        }

    monkeypatch.setattr(openai_compat, "generate_local_batch", fake_batch)
    response = _client(openai_compat.router).post("/v1/chat/completions", json={
        "model": "tiny",
        "n": 3,
        "max_tokens": 64,
        "temperature": 0.7,
        "seed": 42,
        "messages": [
            {"role": "system", "content": "Keep answers concise."},
            {"role": "user", "content": "Summarize the attached design."},
            {"role": "assistant", "content": "I will review the design."},
            {"role": "user", "content": "Focus on the runtime."},
        ],
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2]
    assert [choice["message"]["content"] for choice in body["choices"]] == ["choice A", "choice B", "choice C"]
    assert [choice["finish_reason"] for choice in body["choices"]] == ["length", "stop", "stop"]
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
    assert captured["model_path"] == "/trusted/tiny"
    assert captured["prompts"] == ["Focus on the runtime."] * 3
    assert captured["chat_messages"] == [
        {"role": "system", "content": "Keep answers concise."},
        {"role": "user", "content": "Summarize the attached design."},
        {"role": "assistant", "content": "I will review the design."},
        {"role": "user", "content": "Focus on the runtime."},
    ]
    assert captured["seed"] == 42


def test_openai_n_is_bounded_and_streaming_multi_choice_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(openai_compat, "_native_models", lambda: [{
        "name": "tiny", "path": "/trusted/tiny", "model_type": "llama",
    }])
    client = _client(openai_compat.router)
    too_many = client.post("/v1/chat/completions", json={
        "model": "tiny", "n": 9, "messages": [{"role": "user", "content": "hello"}],
    })
    streamed = client.post("/v1/chat/completions", json={
        "model": "tiny", "n": 2, "stream": True, "messages": [{"role": "user", "content": "hello"}],
    })
    assert too_many.status_code == 422
    assert streamed.status_code == 422
    assert "stream=false" in streamed.json()["error"]["message"]


def test_openai_ollama_multi_choice_is_rejected_without_invoking_ollama(monkeypatch) -> None:
    monkeypatch.setattr(openai_compat, "_resolve_ollama_model", lambda _model: "tiny")
    monkeypatch.setattr(openai_compat, "run_local_chat", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")))
    response = _client(openai_compat.router).post("/v1/chat/completions", json={
        "model": "ollama/tiny", "n": 2, "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 422
    assert "AXIOM-native models only" in response.json()["error"]["message"]
