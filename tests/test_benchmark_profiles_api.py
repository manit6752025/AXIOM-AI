from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import engine as engine_api


def _report():
    keys = (
        "total_wall_time_ms", "measured_wall_time_ms", "median_elapsed_ms", "p95_elapsed_ms",
        "median_prefill_ms", "median_decode_ms", "median_time_to_first_token_ms",
        "median_generation_tokens_per_second", "completion_tokens", "prompt_tokens",
        "aggregate_tokens_per_second",
    )
    return {
        "schema_version": 1, "engine": "axiom-native-engine", "model": "qwen-test",
        "architecture": "qwen3", "model_type": "qwen3", "device": "cpu",
        "precision": "fp32", "memory_mode": "full", "checkpoint_format": "safetensors",
        "warmup_runs_requested": 1, "warmup_runs_completed": 1,
        "iterations_requested": 1, "iterations_completed": 1, "max_new_tokens": 16,
        "sampling": {"temperature": 0.0, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "seed": 1},
        "metrics": {key: 10.0 for key in keys},
        "runs": [{"iteration": 1, "elapsed_ms": 10.0, "prefill_ms": 2.0, "decode_ms": 3.0,
                  "time_to_first_token_ms": 2.0, "generation_tokens_per_second": 20.0,
                  "prompt_tokens": 4, "completion_tokens": 8, "stop_reason": "max_new_tokens"}],
        "memory_stats": {"layer_weight_bytes_staged": 0},
        "generated_at": "2026-11-21T00:00:00+00:00",
    }


def test_native_benchmark_endpoint_saves_profile_only_when_requested(monkeypatch, tmp_path):
    saved = []
    class FakeStore:
        def create(self, name, report, prompt, model_path, *, adapter_used=False):
            saved.append((name, prompt, model_path, adapter_used))
            return {"id": "a" * 32, "name": name, "fingerprint": "b" * 64}
    monkeypatch.setattr(engine_api, "BenchmarkProfileStore", FakeStore)
    monkeypatch.setattr(engine_api, "resolve_local_model_directory", lambda value, _label: value)
    monkeypatch.setattr(engine_api, "benchmark_local", lambda *_args, **_kwargs: _report())
    app = FastAPI()
    app.include_router(engine_api.router)
    response = TestClient(app).post("/api/engine/benchmark", json={
        "model_path": str(tmp_path), "prompt": "private prompt", "profile_name": "Qwen baseline",
    })
    assert response.status_code == 200, response.text
    assert response.json()["profile_record"]["name"] == "Qwen baseline"
    assert saved == [("Qwen baseline", "private prompt", str(tmp_path), False)]


def test_native_benchmark_endpoint_does_not_create_profile_by_default(monkeypatch, tmp_path):
    class FakeStore:
        def __init__(self):
            raise AssertionError("The profile store should not be opened without a profile name.")
    monkeypatch.setattr(engine_api, "BenchmarkProfileStore", FakeStore)
    monkeypatch.setattr(engine_api, "resolve_local_model_directory", lambda value, _label: value)
    monkeypatch.setattr(engine_api, "benchmark_local", lambda *_args, **_kwargs: _report())
    app = FastAPI()
    app.include_router(engine_api.router)
    response = TestClient(app).post("/api/engine/benchmark", json={
        "model_path": str(tmp_path), "prompt": "private prompt",
    })
    assert response.status_code == 200, response.text
    assert response.json()["profile_record"] is None


def test_native_benchmark_profile_routes_list_and_compare_workspace_records(monkeypatch):
    listing = {"schema_version": 1, "total": 1, "profiles": [{"id": "a" * 32, "name": "baseline"}]}
    comparison = {"schema_version": 1, "comparable": True, "match": {"hardware": True}, "metrics": {}}
    class FakeStore:
        def list(self, limit):
            assert limit == 10
            return listing
        def compare(self, left_id, right_id):
            assert left_id == "a" * 32
            assert right_id == "b" * 32
            return comparison
    monkeypatch.setattr(engine_api, "BenchmarkProfileStore", FakeStore)
    app = FastAPI()
    app.include_router(engine_api.router)
    client = TestClient(app)
    listed = client.get("/api/engine/benchmark-profiles?limit=10")
    compared = client.get("/api/engine/benchmark-profiles/compare", params={"left_id": "a" * 32, "right_id": "b" * 32})
    assert listed.status_code == 200 and listed.json() == listing
    assert compared.status_code == 200 and compared.json() == comparison


def test_native_benchmark_profile_name_is_validated_before_execution(monkeypatch):
    monkeypatch.setattr(engine_api, "resolve_local_model_directory", lambda value, _label: value)
    base = {"model_path": "/trusted/model", "prompt": "prompt"}
    assert engine_api.NativeBenchmarkRequest.model_validate({**base, "profile_name": "Valid profile"}).profile_name == "Valid profile"
    try:
        engine_api.NativeBenchmarkRequest.model_validate({**base, "profile_name": "../escape"})
    except Exception:
        pass
    else:
        raise AssertionError("Unsafe profile name should be rejected by the request model.")
    try:
        engine_api.NativeBenchmarkRequest.model_validate({**base, "profile_name": "x" * 65})
    except Exception:
        pass
    else:
        raise AssertionError("Oversized profile name should be rejected by the request model.")
