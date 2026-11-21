from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import inference
from axiom.core.inference import InferenceRuntimeError
from axiom.core.runtime_traces import clear_runtime_traces, list_runtime_traces


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(inference.router)
    return TestClient(app)


def setup_function() -> None:
    clear_runtime_traces()


def teardown_function() -> None:
    clear_runtime_traces()


def _payload():
    return {
        "model": "qwen2.5:0.5b",
        "prompt": "Return a product object.",
        "num_predict": 128,
        "temperature": 0.1,
        "json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 40},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["name", "count"],
        },
    }


def test_structured_inference_passes_schema_and_returns_validated_json(monkeypatch):
    observed = {}

    def fake_generate(model, prompt, **kwargs):
        observed.update(model=model, prompt=prompt, **kwargs)
        return {
            "model": model,
            "response": '{"name":"AXIOM","count":3}',
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 8,
            "total_duration_ms": 15,
            "generated_at": "2026-11-21T00:00:00Z",
        }

    monkeypatch.setattr(inference, "run_local_inference", fake_generate)
    response = _client().post("/api/inference/generate/structured", json=_payload())

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["structured_output"] == {"name": "AXIOM", "count": 3}
    assert result["structured_output_validated"] is True
    assert result["request_id"] == response.headers["X-Request-ID"]
    assert observed["response_format"]["type"] == "object"
    assert observed["response_format"]["additionalProperties"] is False
    assert observed["num_predict"] == 128
    trace = list_runtime_traces()["traces"][0]
    assert trace["operation"] == "inference.generate.structured"
    assert trace["status"] == "success"
    assert "prompt" not in trace and "response" not in trace


def test_structured_inference_rejects_unsupported_schema_before_runtime(monkeypatch):
    monkeypatch.setattr(
        inference, "run_local_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime must not run")),
    )
    payload = _payload()
    payload["json_schema"]["$ref"] = "https://example.invalid/schema"
    response = _client().post("/api/inference/generate/structured", json=payload)
    assert response.status_code == 422
    assert "Unsupported JSON Schema keyword" in response.text


def test_structured_inference_rejects_boolean_token_count(monkeypatch):
    monkeypatch.setattr(
        inference, "run_local_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime must not run")),
    )
    payload = _payload()
    payload["num_predict"] = True
    response = _client().post("/api/inference/generate/structured", json=payload)
    assert response.status_code == 422


def test_structured_inference_rejects_unbounded_token_request(monkeypatch):
    monkeypatch.setattr(
        inference, "run_local_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime must not run")),
    )
    payload = _payload()
    payload["num_predict"] = 513
    response = _client().post("/api/inference/generate/structured", json=payload)
    assert response.status_code == 422


def test_structured_inference_rejects_model_output_that_does_not_match_schema(monkeypatch):
    monkeypatch.setattr(
        inference, "run_local_inference",
        lambda model, _prompt, **_kwargs: {
            "model": model, "response": '{"name":"AXIOM","count":true}',
            "done": True, "prompt_eval_count": 4, "eval_count": 5,
            "total_duration_ms": 10, "generated_at": "2026-11-21T00:00:00Z",
        },
    )
    response = _client().post("/api/inference/generate/structured", json=_payload())
    assert response.status_code == 502
    assert response.json()["detail"] == "The local model output did not satisfy the requested JSON Schema."
    trace = list_runtime_traces()["traces"][0]
    assert trace["error_code"] == "invalid_structured_output"
    assert "response" not in trace


def test_structured_inference_maps_local_runtime_failure_to_503(monkeypatch):
    monkeypatch.setattr(
        inference, "run_local_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(InferenceRuntimeError("local backend offline")),
    )
    response = _client().post("/api/inference/generate/structured", json=_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "Local structured inference runtime did not complete the request."
