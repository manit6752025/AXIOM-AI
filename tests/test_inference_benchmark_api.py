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


def test_benchmark_endpoint_forwards_controls_and_returns_measurements(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected = {
        "schema_version": 1,
        "backend": "ollama",
        "model": "test-model",
        "configuration": {"warmup_runs": 1, "iterations": 2, "concurrency": 2, "measured_requests": 4},
        "measurements": {"wall_time_ms": 234.5, "requests_per_second": 17.0},
        "privacy": {"prompt_returned": False, "responses_returned": False, "results_persisted": False},
        "scope_note": "measured local run",
    }

    def fake_benchmark(model: str, prompt: str, **kwargs) -> dict:
        captured.update({"model": model, "prompt": prompt, **kwargs})
        return expected

    monkeypatch.setattr(inference, "run_local_inference_benchmark", fake_benchmark)
    response = _client().post(
        "/api/inference/benchmark",
        json={
            "model": "test-model",
            "prompt": "measure this",
            "num_predict": 64,
            "temperature": 0.4,
            "warmup_runs": 1,
            "iterations": 2,
            "concurrency": 2,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert {key: value for key, value in result.items() if key != "request_id"} == expected
    assert result["request_id"] == response.headers["X-Request-ID"]
    assert len(result["request_id"]) == 32
    trace = list_runtime_traces()["traces"][0]
    assert trace["request_id"] == result["request_id"]
    assert trace["operation"] == "inference.benchmark"
    assert trace["status"] == "success"
    assert "prompt" not in trace and "response" not in trace
    assert captured == {
        "model": "test-model",
        "prompt": "measure this",
        "num_predict": 64,
        "temperature": 0.4,
        "warmup_runs": 1,
        "iterations": 2,
        "concurrency": 2,
    }


def test_benchmark_endpoint_rejects_excess_total_requests(monkeypatch) -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("invalid benchmark request reached the runtime")

    monkeypatch.setattr(inference, "run_local_inference_benchmark", should_not_run)
    response = _client().post(
        "/api/inference/benchmark",
        json={"model": "test-model", "prompt": "hello", "iterations": 3, "concurrency": 4},
    )

    assert response.status_code == 422
    assert "may not exceed 10" in response.json()["detail"]
    assert len(response.headers["X-Request-ID"]) == 32
    trace = list_runtime_traces()["traces"][0]
    assert trace["request_id"] == response.headers["X-Request-ID"]
    assert trace["status"] == "error"
    assert trace["error_code"] == "request_limit_exceeded"


def test_benchmark_endpoint_maps_runtime_unavailability_to_503(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise InferenceRuntimeError("Local runtime unavailable.")

    monkeypatch.setattr(inference, "run_local_inference_benchmark", unavailable)
    response = _client().post(
        "/api/inference/benchmark",
        json={"model": "test-model", "prompt": "hello", "iterations": 1, "concurrency": 1},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Local inference benchmark failed because the runtime did not complete the request."
    assert response.headers["X-Request-ID"] == list_runtime_traces()["traces"][0]["request_id"]
    assert list_runtime_traces()["traces"][0]["error_code"] == "runtime_unavailable"


def test_benchmark_endpoint_rejects_remote_model_urls() -> None:
    response = _client().post(
        "/api/inference/benchmark",
        json={"model": "https://example.com/model", "prompt": "hello"},
    )

    assert response.status_code == 422
