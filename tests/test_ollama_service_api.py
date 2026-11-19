from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import inference
from axiom.core.ollama_service import OllamaServiceError


def _client(client=("testclient", 50000)) -> TestClient:
    app = FastAPI()
    app.include_router(inference.router)
    return TestClient(app, client=client)


def test_status_uses_managed_runtime_report(monkeypatch):
    expected = {
        "backend": "ollama", "endpoint": "http://127.0.0.1:11434",
        "runtime_installed": True, "status": "offline", "models": [],
        "error": "offline", "checked_at": "2026-10-24T00:00:00+00:00",
        "service_managed_by_axiom": False, "service_can_start": True,
        "service_can_stop": False, "service_control_scope": "loopback_only",
        "service_message": "ready",
    }
    monkeypatch.setattr(inference, "managed_service_status", lambda: expected)

    response = _client().get("/api/inference/status")

    assert response.status_code == 200
    assert response.json() == expected


def test_start_service_maps_startup_errors_to_actionable_response(monkeypatch):
    monkeypatch.setattr(
        inference, "start_service",
        lambda: (_ for _ in ()).throw(OllamaServiceError("Ollama binary is missing.")),
    )

    response = _client().post("/api/inference/service/start", json={})

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama binary is missing."


def test_stop_service_maps_unowned_process_refusal_to_conflict(monkeypatch):
    monkeypatch.setattr(
        inference, "stop_service",
        lambda: (_ for _ in ()).throw(OllamaServiceError("Will not stop an unowned service.")),
    )

    response = _client().post("/api/inference/service/stop", json={})

    assert response.status_code == 409
    assert "unowned" in response.json()["detail"]


def test_model_pull_endpoint_returns_a_bounded_job(monkeypatch):
    expected = {
        "id": "a" * 32, "model": "qwen2.5:0.5b", "status": "queued",
        "created_at": "2026-10-24T00:00:00+00:00", "updated_at": "2026-10-24T00:00:00+00:00",
        "progress": "Queued", "percent": None, "logs": [], "error": None,
        "cancel_requested": False,
    }
    captured = {}
    def fake_start(model):
        captured["model"] = model
        return expected
    monkeypatch.setattr(inference, "start_model_pull", fake_start)

    response = _client().post("/api/inference/models/pull", json={"model": "qwen2.5:0.5b"})

    assert response.status_code == 200
    assert response.json() == expected
    assert captured["model"] == "qwen2.5:0.5b"


def test_model_pull_rejects_remote_urls_before_queueing(monkeypatch):
    monkeypatch.setattr(inference, "start_model_pull", lambda *_args: AssertionError("must not queue a URL"))

    response = _client().post("/api/inference/models/pull", json={"model": "https://example.com/model"})

    assert response.status_code == 422


def test_runtime_management_rejects_nonlocal_clients_by_default(monkeypatch):
    monkeypatch.delenv("AXIOM_ALLOW_REMOTE_RUNTIME_MANAGEMENT", raising=False)
    monkeypatch.setattr(inference, "start_service", lambda: (_ for _ in ()).throw(AssertionError("must be blocked")))

    response = _client(client=("192.168.1.42", 50000)).post("/api/inference/service/start", json={})

    assert response.status_code == 403


def test_model_inspection_endpoint_returns_only_metadata(monkeypatch):
    expected = {
        "schema_version": 1, "model": "qwen2.5:0.5b", "installed": True,
        "size_bytes": 123, "modified_at": None, "format": "gguf", "family": "qwen2",
        "families": ["qwen2"], "parameter_size": "0.5B", "parameter_count": 500000000,
        "quantization_level": "Q4_K_M", "context_length": 32768, "capabilities": ["completion"],
        "template_configured": True, "license_present": True, "system_prompt_configured": False,
        "scope_note": "metadata only",
    }
    monkeypatch.setattr(inference, "inspect_local_model", lambda _model: expected)
    response = _client().post("/api/inference/models/inspect", json={"model": "qwen2.5:0.5b"})
    assert response.status_code == 200
    assert response.json() == expected


def test_model_unload_endpoint_reports_downloads_are_preserved(monkeypatch):
    expected = {
        "schema_version": 1, "model": "qwen2.5:0.5b", "unloaded": True,
        "model_files_deleted": False, "scope_note": "memory only",
    }
    monkeypatch.setattr(inference, "unload_local_model", lambda _model: expected)
    response = _client().post("/api/inference/models/unload", json={"model": "qwen2.5:0.5b"})
    assert response.status_code == 200
    assert response.json()["model_files_deleted"] is False


def test_running_model_endpoint_returns_point_in_time_memory_report(monkeypatch):
    expected = {
        "schema_version": 1, "backend": "ollama", "endpoint": "http://127.0.0.1:11434",
        "models": [{"name": "qwen2.5:0.5b", "size_bytes": 800, "size_vram_bytes": 300,
                    "size_ram_bytes": 500, "memory_placement": "mixed", "expires_at": None,
                    "digest_prefix": None, "format": "gguf", "family": "qwen2",
                    "parameter_size": "0.5B", "quantization_level": "Q4_K_M"}],
        "count": 1, "scope_note": "Runtime-reported memory",
    }
    monkeypatch.setattr(inference, "list_running_local_models", lambda: expected)
    response = _client().get("/api/inference/models/running")
    assert response.status_code == 200
    assert response.json() == expected


def test_streaming_inference_emits_ndjson_token_deltas_and_request_id(monkeypatch):
    events = [
        {"type": "token", "token": "Hello "},
        {"type": "token", "token": "from Ollama"},
        {"type": "complete", "result": {
            "model": "qwen2.5:0.5b", "response": "Hello from Ollama", "done": True,
            "done_reason": "stop", "prompt_eval_count": 4, "eval_count": 3,
            "total_duration_ms": 8.0, "generated_at": "2026-10-24T00:00:00+00:00",
            "execution": "local",
        }},
    ]
    observed: dict[str, object] = {}

    def fake_stream(*args, **kwargs):
        observed.update(kwargs)
        return iter(events)

    monkeypatch.setattr(inference, "stream_local_inference", fake_stream)
    monkeypatch.setattr(inference, "record_runtime_trace", lambda **_kwargs: None)

    response = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Say hello", "num_predict": 32, "temperature": 0.2,
        "min_p": 0.2,
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert observed["min_p"] == 0.2
    assert response.headers.get("x-request-id")
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert [payload["type"] for payload in payloads] == ["token", "token", "complete"]
    assert [payload["token"] for payload in payloads[:2]] == ["Hello ", "from Ollama"]
    assert payloads[-1]["result"]["response"] == ""
    assert payloads[-1]["result"]["request_id"] == response.headers["x-request-id"]


def test_streaming_inference_sanitizes_runtime_failures(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("sensitive /home/test/private traceback")
        yield {}
    monkeypatch.setattr(inference, "stream_local_inference", broken)
    monkeypatch.setattr(inference, "record_runtime_trace", lambda **_kwargs: None)

    response = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Say hello",
    })

    assert response.status_code == 200
    assert "sensitive" not in response.text
    assert "/home/test/private" not in response.text
    payloads = [__import__("json").loads(line) for line in response.text.splitlines() if line.strip()]
    assert payloads[-1]["type"] == "error"
    assert payloads[-1]["message"] == "Local Ollama stream failed unexpectedly."


def test_streaming_api_passes_advanced_sampling_options_to_local_runtime(monkeypatch):
    captured = {}
    def fake_stream(_model, _prompt, **kwargs):
        captured.update(kwargs)
        return iter([
            {"type": "token", "token": "configured"},
            {"type": "complete", "result": {
                "model": "qwen2.5:0.5b", "response": "configured", "done": True,
                "done_reason": "stop", "prompt_eval_count": 2, "eval_count": 1,
                "total_duration_ms": 2, "generated_at": "2026-10-24T00:00:00+00:00",
                "execution": "local",
            }},
        ])
    monkeypatch.setattr(inference, "stream_local_inference", fake_stream)
    monkeypatch.setattr(inference, "record_runtime_trace", lambda **_kwargs: None)

    response = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Hello", "top_p": 0.83, "top_k": 36,
        "min_p": 0.2, "repetition_penalty": 1.2, "seed": 123, "num_ctx": 8192,
        "stop_sequences": ["DONE", "END"],
    })

    assert response.status_code == 200
    assert captured == {
        "num_predict": 256, "temperature": 0.7, "top_p": 0.83, "top_k": 36, "min_p": 0.2,
        "repetition_penalty": 1.2, "seed": 123, "num_ctx": 8192,
        "stop_sequences": ["DONE", "END"], "cancel_event": captured["cancel_event"],
        "on_response": captured["on_response"],
    }


def test_generate_api_rejects_oversized_context_and_stop_sequence_lists():
    too_large_context = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Hello", "num_ctx": 999999,
    })
    too_many_stops = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Hello", "stop_sequences": [str(i) for i in range(17)],
    })
    min_p_below_range = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Hello", "min_p": -0.01,
    })
    min_p_above_range = _client().post("/api/inference/generate/stream", json={
        "model": "qwen2.5:0.5b", "prompt": "Hello", "min_p": 1.01,
    })
    assert too_large_context.status_code == 422
    assert too_many_stops.status_code == 422
    assert min_p_below_range.status_code == 422
    assert min_p_above_range.status_code == 422
