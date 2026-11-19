from __future__ import annotations

from typing import Any

import pytest

from axiom.core import llm_client


def test_provider_catalog_lists_local_and_cloud_adapters_without_credentials() -> None:
    result = llm_client.provider_catalog()
    providers = {item["id"]: item for item in result["providers"]}

    assert {"axiom-local", "ollama-local", "openai", "anthropic", "gemini", "deepseek", "moonshot", "openrouter", "openai-compatible"} <= providers.keys()
    assert providers["axiom-local"]["privacy"] == "local-only"
    assert providers["deepseek"]["requires_api_key"] is True
    assert result["credentials_persisted"] is False


def test_local_model_inventory_uses_only_native_compatible_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "discover_local_engine_assets",
        lambda: {"models": [
            {"name": "local-llama", "path": "/trusted/local-llama", "model_type": "llama", "architecture": "LlamaForCausalLM"},
            {"name": "local-gguf", "path": "/trusted/local-gguf", "model_type": "gguf", "architecture": None},
        ]},
    )

    result = llm_client.list_models("axiom-local")

    assert result["models"] == ["local-llama"]
    assert result["execution"] == "local"
    assert result["credential_storage"] == "not_applicable"


def test_local_chat_uses_axiom_native_stream_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "_native_models",
        lambda: [{"name": "tiny-llama", "path": "/trusted/tiny-llama", "model_type": "llama"}],
    )
    captured: dict[str, Any] = {}

    def fake_generate(model_path: str, prompt: str, **kwargs: Any):
        captured.update({"model_path": model_path, "prompt": prompt, **kwargs})
        return iter([{
            "type": "complete",
            "result": {
                "response": "Local result",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "device": "cpu",
                "precision": "fp16",
                "precision_selection": {"requested": "fp16", "selected": "fp16"},
            },
        }])

    monkeypatch.setattr(llm_client, "generate_local_stream", fake_generate)
    result = llm_client.chat(
        "axiom-local",
        "tiny-llama",
        [{"role": "user", "content": "Hello"}],
        max_tokens=7,
        temperature=0,
        top_p=0.82,
        top_k=40,
        min_p=0.2,
        repetition_penalty=1.15,
        seed=123,
        stop_sequences=["END", "END"],
        precision="fp16",
    )

    assert result["response"] == "Local result"
    assert result["execution"] == "local"
    assert result["device"] == "cpu"
    assert result["precision"] == "fp16"
    assert result["precision_selection"] == {"requested": "fp16", "selected": "fp16"}
    assert captured["precision"] == "fp16"
    assert captured["model_path"] == "/trusted/tiny-llama"
    assert captured["chat_messages"] == [{"role": "user", "content": "Hello"}]
    assert captured["max_new_tokens"] == 7
    assert captured["top_p"] == 0.82
    assert captured["top_k"] == 40
    assert captured["min_p"] == 0.2
    assert captured["repetition_penalty"] == 1.15
    assert captured["seed"] == 123
    assert captured["stop_sequences"] == ("END",)


def test_cloud_chat_requires_explicit_consent_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_request(*_args: Any, **_kwargs: Any):
        pytest.fail("No provider request is allowed without explicit consent.")

    monkeypatch.setattr(llm_client, "_provider_request", unexpected_request)

    with pytest.raises(ValueError, match="Explicitly confirm"):
        llm_client.chat(
            "deepseek",
            "deepseek-chat",
            [{"role": "user", "content": "private prompt"}],
            api_key="sentinel-secret",
            consent=False,
        )


def test_deepseek_openai_compatible_chat_maps_response_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(provider: str, api_key: str, endpoint: str, payload=None, *, base_url=None):
        captured.update({"provider": provider, "api_key": api_key, "endpoint": endpoint, "payload": payload, "base_url": base_url})
        return {
            "choices": [{"message": {"role": "assistant", "content": "Answer"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        }

    monkeypatch.setattr(llm_client, "_provider_request", fake_request)
    result = llm_client.chat(
        "deepseek",
        "deepseek-chat",
        [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Question"}],
        api_key="secret-sentinel",
        consent=True,
        max_tokens=32,
        temperature=0.2,
    )

    assert result["response"] == "Answer"
    assert result["prompt_tokens"] == 9
    assert result["completion_tokens"] == 4
    assert result["credential_storage"] == "not_saved"
    assert result["execution"] == "remote"
    assert captured["endpoint"] == "/chat/completions"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Question"},
    ]
    assert "secret-sentinel" not in str(result)


def test_provider_specific_anthropic_payload_keeps_system_out_of_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(provider: str, api_key: str, endpoint: str, payload=None, *, base_url=None):
        captured.update({"provider": provider, "endpoint": endpoint, "payload": payload})
        return {
            "content": [{"type": "text", "text": "Claude response"}],
            "usage": {"input_tokens": 11, "output_tokens": 8},
        }

    monkeypatch.setattr(llm_client, "_provider_request", fake_request)
    result = llm_client.chat(
        "anthropic",
        "claude-test",
        [{"role": "system", "content": "Carefully answer"}, {"role": "user", "content": "Explain"}],
        api_key="unused-in-test",
        consent=True,
    )

    assert result["response"] == "Claude response"
    assert result["prompt_tokens"] == 11
    assert captured["endpoint"] == "/messages"
    assert captured["payload"]["system"] == "Carefully answer"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Explain"}]


def test_gemini_model_listing_normalizes_model_names() -> None:
    model_ids = llm_client._model_ids("gemini", {
        "models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-only", "supportedGenerationMethods": ["embedContent"]},
        ],
    })
    assert model_ids == ["gemini-2.5-flash"]


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "assistant", "content": "No user input"}],
    [{"role": "user", "content": "   "}],
])
def test_chat_rejects_empty_or_userless_conversations(messages: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError):
        llm_client._validate_messages(messages)


def test_chat_rejects_combined_message_budget_overflow() -> None:
    messages = [{"role": "user", "content": "x" * 12_000}, {"role": "assistant", "content": "y" * 12_000}, {"role": "user", "content": "z"}]
    with pytest.raises(ValueError, match="Combined chat content"):
        llm_client._validate_messages(messages)


def test_openrouter_uses_the_fixed_public_api_host() -> None:
    assert llm_client._base_url("openrouter", None) == "https://openrouter.ai/api/v1"


def test_custom_provider_requires_public_https_v1_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "validate_remote_base_url", lambda value: value.rstrip("/"))
    result = llm_client._base_url("openai-compatible", "https://public.example/api/v1/")
    assert result == "https://public.example/api/v1"
    with pytest.raises(ValueError, match="custom base URL"):
        llm_client._base_url("deepseek", "https://other.example/v1")


def test_public_address_pinning_rejects_private_dns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    monkeypatch.setattr(
        llm_client.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(llm_client.LLMClientError, match="non-public address"):
        llm_client._resolve_public_addresses("https://provider.example/v1")


def test_custom_provider_uses_pinned_public_ip_and_never_follows_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status = 200

        def read(self, limit: int) -> bytes:
            captured["response_limit"] = limit
            return b'{"data":[{"id":"remote-test-model"}]}'

    class FakeConnection:
        def __init__(self, host: str, **kwargs: Any) -> None:
            captured["host"] = host
            captured["connection_options"] = kwargs

        def request(self, method: str, path: str, *, body: bytes | None, headers: dict[str, str]) -> None:
            captured.update({"method": method, "path": path, "body": body, "headers": headers})

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(llm_client, "validate_remote_base_url", lambda value: value.rstrip("/"))
    monkeypatch.setattr(llm_client, "_resolve_public_addresses", lambda _base: ("93.184.216.34",))
    monkeypatch.setattr(llm_client, "_PinnedHTTPSConnection", FakeConnection)

    response = llm_client._provider_request(
        "openai-compatible",
        "secret-sentinel",
        "/models",
        base_url="https://provider.example/api/v1",
    )

    assert response == {"data": [{"id": "remote-test-model"}]}
    assert captured["host"] == "provider.example"
    assert captured["connection_options"]["addresses"] == ("93.184.216.34",)
    assert captured["path"] == "/api/v1/models"
    assert captured["method"] == "GET"
    assert captured["headers"]["Authorization"] == "Bearer secret-sentinel"
    assert captured["closed"] is True
    assert "secret-sentinel" not in str(response)


def test_ollama_local_provider_lists_only_local_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "ollama_runtime_status", lambda: {
        "backend": "ollama", "status": "online", "models": ["qwen2.5:0.5b"], "error": None,
    })
    result = llm_client.list_models("ollama-local")

    assert result["models"] == ["qwen2.5:0.5b"]
    assert result["execution"] == "local"
    assert result["credential_storage"] == "not_applicable"


def test_ollama_local_provider_sends_structured_turns_to_loopback_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    expected_messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Previous local answer"},
        {"role": "user", "content": "Follow up"},
    ]

    def fake_chat(
        model: str, messages: list[dict[str, str]], *, num_predict: int, temperature: float,
        top_p: float, top_k: int, min_p: float, repetition_penalty: float, seed: int | None, stop_sequences,
    ):
        captured.update({
            "model": model, "messages": messages, "num_predict": num_predict, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "min_p": min_p, "repetition_penalty": repetition_penalty,
            "seed": seed, "stop_sequences": stop_sequences,
        })
        return {
            "model": model, "response": "Ollama answer", "prompt_eval_count": 7,
            "eval_count": 3, "generated_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(llm_client, "run_local_chat", fake_chat)
    result = llm_client.chat(
        "ollama-local", "qwen2.5:0.5b", expected_messages,
        max_tokens=18, temperature=0.2, top_p=0.85, top_k=32,
        min_p=0.2, repetition_penalty=1.2, seed=7, stop_sequences=["DONE"], consent=False,
    )

    assert result["provider"] == "ollama-local"
    assert result["response"] == "Ollama answer"
    assert result["execution"] == "local"
    assert result["credential_storage"] == "not_applicable"
    assert result["prompt_tokens"] == 7
    assert captured == {
        "model": "qwen2.5:0.5b",
        "messages": expected_messages,
        "num_predict": 18,
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 32,
        "min_p": 0.2,
        "repetition_penalty": 1.2,
        "seed": 7,
        "stop_sequences": ("DONE",),
    }


@pytest.mark.parametrize("options", [
    {"top_p": 0},
    {"top_p": float("nan")},
    {"top_k": -1},
    {"top_k": True},
    {"min_p": -0.01},
    {"min_p": 1.01},
    {"min_p": float("nan")},
    {"min_p": True},
    {"repetition_penalty": 0},
    {"seed": -1},
    {"seed": True},
    {"stop_sequences": [""]},
    {"stop_sequences": ["x" * 257]},
    {"stop_sequences": ["STOP"] * 17},
])
def test_local_sampling_options_are_validated_before_dispatch(monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]) -> None:
    monkeypatch.setattr(llm_client, "run_local_chat", lambda *_args, **_kwargs: pytest.fail("invalid input must not dispatch"))
    with pytest.raises(ValueError):
        llm_client.chat("ollama-local", "qwen2.5:0.5b", [{"role": "user", "content": "test"}], **options)


def test_cloud_provider_rejects_unsupported_local_only_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "_provider_request", lambda *_args, **_kwargs: pytest.fail("unsupported parameters must not be sent"))
    with pytest.raises(ValueError, match="local AXIOM Native Engine and Ollama"):
        llm_client.chat(
            "deepseek", "deepseek-chat", [{"role": "user", "content": "private"}],
            api_key="sentinel-secret", consent=True, top_k=25, min_p=0.2,
        )


def test_native_model_inventory_excludes_ambiguous_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "discover_local_engine_assets", lambda: {
        "models": [
            {"name": "qwen-duplicate", "path": "/trusted/one", "model_type": "qwen2"},
            {"name": "qwen-duplicate", "path": "/trusted/two", "model_type": "qwen2"},
            {"name": "llama-unique", "path": "/trusted/three", "model_type": "llama"},
            {"name": "unsupported-architecture", "path": "/trusted/four", "model_type": "other"},
        ],
    })
    assert [item["name"] for item in llm_client._native_models()] == ["llama-unique"]
