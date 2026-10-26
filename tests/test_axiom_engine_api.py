from __future__ import annotations

import json
from threading import Event
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from axiom.api import engine as engine_api
from axiom.api import openai_compat


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(engine_api.router)
    app.include_router(openai_compat.router)
    return TestClient(app)


def test_engine_request_accepts_absolute_paths_only_inside_explicit_roots(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    request = engine_api.GenerateRequest(model_path=str(model), prompt="hello")
    assert request.memory_mode == "auto"
    assert request.precision == "auto"
    assert engine_api.GenerateRequest(model_path=str(model), prompt="hello", precision="fp16").precision == "fp16"
    with pytest.raises(ValidationError, match="precision"):
        engine_api.GenerateRequest(model_path=str(model), prompt="hello", precision="int4")

    assert request.model_path == str(model.resolve())


def test_native_benchmark_api_forwards_bounded_local_options_without_response_payload(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "models"
    model = root / "tiny"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))
    captured = {}

    def fake_benchmark(model_path, prompt, **kwargs):
        captured.update(model_path=model_path, prompt=prompt, **kwargs)
        return {
            "schema_version": 1,
            "iterations_completed": 2,
            "warmups_excluded": True,
            "metrics": {"median_elapsed_ms": 12.0},
            "privacy": {
                "prompt_returned": False, "response_returned": False,
                "results_persisted": False, "cloud_provider_used": False,
            },
        }

    monkeypatch.setattr(engine_api, "benchmark_local", fake_benchmark)
    response = _client().post("/api/engine/benchmark", json={
        "model_path": str(model),
        "prompt": "private benchmark prompt",
        "warmup_runs": 1,
        "iterations": 2,
        "max_new_tokens": 8,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 4,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stop_sequences": ["END"],
        "device": "cpu",
        "memory_mode": "layer_streaming",
        "precision": "bf16",
    })
    assert response.status_code == 200, response.text
    assert response.json()["warmups_excluded"] is True
    assert "response" not in response.json()
    assert isinstance(captured.pop("cancel_event"), Event)
    assert captured == {
        "model_path": str(model.resolve()),
        "prompt": "private benchmark prompt",
        "adapter_path": None,
        "warmup_runs": 1,
        "iterations": 2,
        "max_new_tokens": 8,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 4,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stop_sequences": ["END"],
        "device_preference": "cpu",
        "memory_mode": "layer_streaming",
        "precision": "bf16",
    }


def test_native_benchmark_api_rejects_iterations_or_output_over_the_limit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))
    monkeypatch.setattr(
        engine_api,
        "benchmark_local",
        lambda *_args, **_kwargs: pytest.fail("invalid benchmark must be rejected before execution"),
    )
    for invalid in ({"iterations": 6}, {"max_new_tokens": 129}, {"warmup_runs": 3}):
        response = _client().post("/api/engine/benchmark", json={
            "model_path": str(model), "prompt": "hello", **invalid,
        })
        assert response.status_code == 422, response.text


def test_batch_generate_request_validates_size_prompt_limits_and_sampling(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny"
    model.mkdir(parents=True)
    # This endpoint-schema test still needs a directory discoverable by the
    # trusted-root validator; weights are deliberately not loaded by this test.
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    request = engine_api.BatchGenerateRequest(
        model_path=str(model), prompts=["first prompt", "second prompt"],
        max_new_tokens=17, temperature=0, top_p=0.8, top_k=4, min_p=0.1,
        repetition_penalty=1.1, seed=42, stop_sequences=["END", "END"],
    )
    assert request.model_path == str(model.resolve())
    assert request.prompts == ["first prompt", "second prompt"]
    assert request.top_k == 4
    assert request.stop_sequences == ["END"]

    with pytest.raises(ValidationError, match="at least 1 item"):
        engine_api.BatchGenerateRequest(model_path=str(model), prompts=[])
    with pytest.raises(ValidationError, match="non-whitespace text"):
        engine_api.BatchGenerateRequest(model_path=str(model), prompts=[" ", "ok"])
    with pytest.raises(ValidationError, match="Combined batch prompts"):
        engine_api.BatchGenerateRequest(model_path=str(model), prompts=["a" * 12000, "b" * 12000, "c"])
    with pytest.raises(ValidationError, match="too_long"):
        engine_api.BatchGenerateRequest(model_path=str(model), prompts=["x"] * 129)


def test_native_batch_generate_endpoint_dispatches_to_axiom_batch_runner(monkeypatch) -> None:
    monkeypatch.setattr(engine_api, "resolve_local_model_directory", lambda value, label: value)
    expected = {
        "engine": "axiom-native-engine",
        "execution_backend": "pytorch-native-axiom",
        "batch_size": 2,
        "results": [
            {"index": 1, "response": "one", "completion_tokens": 1, "stop_reason": "max_new_tokens"},
            {"index": 2, "response": "two", "completion_tokens": 1, "stop_reason": "eos_token"},
        ],
        "local_only": True,
    }
    observed = {}

    def fake_batch(model_path, prompts, **kwargs):
        observed.update(model_path=model_path, prompts=prompts, **kwargs)
        return expected

    monkeypatch.setattr(engine_api, "generate_local_batch", fake_batch)
    response = _client().post("/api/engine/generate/batch", json={
        "model_path": "/trusted/models/tiny",
        "prompts": ["first", "second"],
        "max_new_tokens": 32,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 4,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stop_sequences": ["END"],
        "device": "cpu",
        "memory_mode": "layer_streaming",
        "precision": "bf16",
    })
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert {key: value for key, value in observed.items() if key != "cancel_event"} == {
        "model_path": "/trusted/models/tiny",
        "prompts": ["first", "second"],
        "adapter_path": None,
        "max_new_tokens": 32,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 4,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stop_sequences": ["END"],
        "device_preference": "cpu",
        "memory_mode": "layer_streaming",
        "precision": "bf16",
    }
    assert isinstance(observed["cancel_event"], Event)


def test_engine_request_rejects_absolute_path_outside_configured_roots(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValidationError, match="checkpoint directory|trusted model root"):
        engine_api.GenerateRequest(model_path=str(outside), prompt="hello")


def test_engine_request_rejects_symlink_escape(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = root / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable on this platform: {exc}")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValidationError, match="symlink|trusted model root|checkpoint directory"):
        engine_api.GenerateRequest(model_path=str(escape), prompt="hello")


def test_engine_telemetry_endpoint_returns_privacy_bounded_snapshot(monkeypatch) -> None:
    expected = {
        "schema_version": 1,
        "sampled_at": "2026-10-24T04:00:00+00:00",
        "sample_type": "point_in_time",
        "cpu": {"logical_cores": 8, "system_percent": 12.5, "process_percent": 1.0},
        "memory": {"total_bytes": 1000, "available_bytes": 400, "used_bytes": 600, "percent": 60.0},
        "process": {"pid": 1, "rss_bytes": 200, "threads": 4},
        "devices": [],
        "loaded_runtime": {"loaded": False},
        "disks": [],
        "privacy_scope": {
            "prompt_content_included": False, "generated_content_included": False,
            "model_paths_included": False, "environment_values_included": False,
            "credentials_included": False, "network_access_performed": False,
        },
        "scope_note": "Point-in-time telemetry only.",
    }
    monkeypatch.setattr(engine_api, "runtime_telemetry", lambda: expected)
    response = _client().get("/api/engine/telemetry")
    assert response.status_code == 200
    assert response.json() == expected


def test_engine_unload_endpoint_calls_lifecycle_handler(monkeypatch) -> None:
    expected = {
        "schema_version": 1,
        "unloaded": True,
        "loaded": False,
        "scope_note": "Model unloaded for test.",
    }
    monkeypatch.setattr(engine_api, "unload_local_model", lambda: expected)

    response = _client().post("/api/engine/unload")

    assert response.status_code == 200
    assert response.json() == expected



def test_native_stream_api_emits_ndjson_token_and_completion_events(monkeypatch) -> None:
    monkeypatch.setattr(engine_api, "resolve_local_model_directory", lambda value, label: value)

    observed: dict[str, object] = {}

    def fake_stream(
        model_path, prompt, *, adapter_path=None, max_new_tokens=256, temperature=0.7,
        top_p=0.9, top_k=0, min_p=0.0, repetition_penalty=1.0, seed=None, stop_sequences=(), include_token_events=True,
        cancel_event=None, device_preference="auto", memory_mode="full", precision="auto",
    ):
        observed.update({
            "model_path": model_path, "prompt": prompt, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": min_p,
            "repetition_penalty": repetition_penalty, "seed": seed,
            "stop_sequences": list(stop_sequences),
            "include_token_events": include_token_events,
            "device_preference": device_preference,
            "memory_mode": memory_mode,
            "precision": precision,
        })
        yield {"type": "token", "response": "hi", "completion_tokens": 1}
        yield {
            "type": "complete",
            "result": {
                "engine": "axiom-native-engine",
                "execution_backend": "pytorch-native-axiom",
                "response": "hi",
                "completion_tokens": 1,
                "local_only": True,
            },
        }

    monkeypatch.setattr(engine_api, "generate_local_stream", fake_stream)
    response = _client().post(
        "/api/engine/generate/stream",
        json={
            "model_path": "/trusted/models/tiny",
            "prompt": "hello",
            "max_new_tokens": 3,
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 4,
            "min_p": 0.2,
            "repetition_penalty": 1.1,
            "seed": 42,
            "stop_sequences": ["<END>", "###"],
            "device": "cpu",
            "memory_mode": "layer_streaming",
            "precision": "bf16",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[0] == {"type": "token", "response": "hi", "completion_tokens": 1}
    assert events[1]["type"] == "complete"
    assert events[1]["result"]["execution_backend"] == "pytorch-native-axiom"
    assert observed == {
        "model_path": "/trusted/models/tiny",
        "prompt": "hello",
        "max_new_tokens": 3,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 4,
        "min_p": 0.2,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stop_sequences": ["<END>", "###"],
        "include_token_events": True,
        "device_preference": "cpu",
        "memory_mode": "layer_streaming",
        "precision": "bf16",
    }


def test_generate_request_rejects_invalid_stop_sequence_lists(monkeypatch) -> None:
    from pydantic import ValidationError
    from axiom.api import engine as engine_module
    from axiom.api.engine import GenerateRequest

    monkeypatch.setattr(engine_module, "resolve_local_model_directory", lambda value, label: value)
    base = {"model_path": "/trusted/models/tiny", "prompt": "hello"}
    with pytest.raises(ValidationError, match="device"):
        engine_api.GenerateRequest(**base, device="network")
    assert engine_api.GenerateRequest(**base, memory_mode="disk_streaming").memory_mode == "disk_streaming"
    with pytest.raises(ValidationError, match="memory_mode"):
        engine_api.GenerateRequest(**base, memory_mode="automatic")
    assert engine_api.GenerateRequest(**base).min_p == 0.0
    with pytest.raises(ValidationError, match="min_p"):
        engine_api.GenerateRequest(**base, min_p=1.1)
    with pytest.raises(ValidationError, match="min_p"):
        engine_api.GenerateRequest(**base, min_p=-0.01)
    with pytest.raises(ValidationError, match="stop sequences|Stop sequences"):
        GenerateRequest(**base, stop_sequences=[" "])
    with pytest.raises(ValidationError, match="stop sequence|Stop sequence"):
        GenerateRequest(**base, stop_sequences=["x" * 257])
    with pytest.raises(ValidationError):
        GenerateRequest(**base, stop_sequences=["END"] * 17)


def test_adapter_compatibility_api_returns_per_device_report(monkeypatch) -> None:
    monkeypatch.setattr(engine_api, "resolve_local_model_directory", lambda value, label: value)
    expected = {
        "schema_version": 1,
        "base_model": {"name": "base", "model_type": "llama", "architecture": "LlamaForCausalLM"},
        "adapter": {"name": "lora", "rank": 8, "alpha": 16.0, "target_modules": ["q_proj", "v_proj"]},
        "structurally_compatible": True,
        "compatible_on_any_device": True,
        "adapter_tensor_count": 4,
        "checked_layers": 12,
        "devices": [{"name": "CPU", "available": True, "compatible": True, "status": "compatible", "reason": "Preflight passed."}],
        "issues": [],
        "scope_note": "Metadata only.",
    }
    monkeypatch.setattr(engine_api, "check_local_adapter_compatibility", lambda model_path, adapter_path: expected)
    response = _client().post("/api/engine/adapters/compatibility", json={
        "model_path": "/trusted/models/base",
        "adapter_path": "/trusted/models/lora",
    })

    assert response.status_code == 200
    assert response.json() == expected


def test_quantization_estimator_compares_formats_against_memory_budgets(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny-llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "tie_word_embeddings": False,
    }), encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    response = _client().post("/api/engine/quantization/estimate", json={
        "model_path": str(model),
        "context_length": 16,
        "batch_size": 1,
        "ram_budget_gib": 8,
        "vram_budget_gib": 4,
    })

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["schema_version"] == 1
    assert report["model"]["name"] == "tiny-llama"
    assert report["request"]["context_length"] == 16
    assert report["assumptions"]["kv_cache_context_length"] == 16
    assert report["assumptions"]["kv_prefill_temporary_context_length"] == 0
    candidates = {item["format"]: item for item in report["candidates"]}
    assert set(candidates) == {"int4", "int8", "bf16", "fp16", "fp32"}
    assert candidates["int4"]["estimated_total_gib"] > 0
    assert all(item["estimated_kv_prefill_temporary_gib"] == 0 for item in candidates.values())
    assert candidates["int4"]["estimated_total_gib"] < candidates["int8"]["estimated_total_gib"]
    assert candidates["int8"]["estimated_total_gib"] < candidates["fp16"]["estimated_total_gib"]
    assert report["recommendation"]["format"] == "bf16"
    assert report["recommendation"]["status"] == "fits_budget"
    assert candidates["int8"]["estimated_total_gib"] < candidates["fp16"]["estimated_total_gib"]
    assert candidates["fp16"]["estimated_total_gib"] < candidates["fp32"]["estimated_total_gib"]
    assert candidates["int4"]["native_checkpoint_supported"] is False
    assert candidates["int8"]["native_checkpoint_supported"] is False
    assert candidates["fp16"]["native_checkpoint_supported"] is True
    assert candidates["fp32"]["native_checkpoint_supported"] is True
    assert candidates["int4"]["fits_ram_budget"] is True
    assert candidates["fp32"]["fits_vram_budget"] is True
    assert report["assumptions"]["safety_margin_percent"] == 20
    assert report["assumptions"]["uncertainty_range_percent"] == {"low": -15, "high": 25}
    assert "not predict model quality/speed" in report["scope_note"]
    assert "not inspect tensor contents" in report["scope_note"]


def test_quantization_estimator_reports_no_native_fit_for_small_budget(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "large-llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8192,
        "intermediate_size": 22016,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128000,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }), encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    response = _client().post("/api/engine/quantization/estimate", json={
        "model_path": str(model),
        "context_length": 4096,
        "batch_size": 1,
        "ram_budget_gib": 0.1,
        "vram_budget_gib": 0.1,
    })

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["recommendation"]["format"] is None
    assert report["recommendation"]["status"] == "no_native_candidate_fits"
    assert "No currently native-supported precision" in report["recommendation"]["note"]


def test_quantization_estimator_rejects_context_above_model_limit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny-llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "tie_word_embeddings": False,
    }), encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    response = _client().post("/api/engine/quantization/estimate", json={
        "model_path": str(model),
        "context_length": 33,
        "batch_size": 1,
        "ram_budget_gib": 8,
    })

    assert response.status_code == 422
    assert "exceeds this checkpoint's limit" in response.json()["detail"]


def test_quantization_estimator_requires_at_least_one_budget(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny-llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "tie_word_embeddings": False,
    }), encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    response = _client().post("/api/engine/quantization/estimate", json={
        "model_path": str(model), "context_length": 16, "batch_size": 1,
    })

    assert response.status_code == 422
    assert "Provide a RAM and/or VRAM budget" in response.text


def test_native_model_discovery_lists_local_metadata_without_loading_weights(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny-llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 8,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "bos_token_id": 1,
            "tie_word_embeddings": False,
        }),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"metadata discovery does not load tensor data")
    adapter = root / "tiny-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"task_type":"CAUSAL_LM"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"deferred until load")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    response = _client().get("/api/engine/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["models"] == [{
        "path": str(model.resolve()),
        "name": "tiny-llama",
        "model_type": "llama",
        "architecture": "LlamaForCausalLM",
        "weight_format": "safetensors",
        "ready_for_native_attempt": True,
    }]
    assert payload["adapters"] == [{
        "path": str(adapter.resolve()),
        "name": "tiny-lora",
        "weight_format": "safetensors LoRA",
    }]
    assert "not loaded or fully validated" in payload["scope_note"]



def test_engine_request_rejects_adapter_directory_as_base_model(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"task_type":"CAUSAL_LM"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValidationError, match="checkpoint directory|trusted model root"):
        engine_api.GenerateRequest(model_path=str(adapter), prompt="hello")


def test_engine_request_rejects_base_model_directory_as_adapter(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "base-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValidationError, match="checkpoint directory|trusted model root"):
        engine_api.GenerateRequest(model_path=str(model), adapter_path=str(model), prompt="hello")


def test_engine_stream_endpoint_yields_token_snapshots_and_final_metrics(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "tiny-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))
    completed = {
        "engine": "axiom-native-engine",
        "execution_backend": "pytorch-native-axiom",
        "transformers_generation_used": False,
        "ollama_used": False,
        "model": "tiny-model",
        "architecture": "LlamaForCausalLM",
        "model_type": "llama",
        "adapter": None,
        "device": "cpu",
        "checkpoint_format": "safetensors",
        "decoder": "axiom-autoregressive-kv-cache",
        "attention_backend": "pytorch-scaled-dot-product-attention",
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "seed": None,
        },
        "prefill_ms": 4.2,
        "decode_ms": 5.3,
        "time_to_first_token_ms": 4.2,
        "generation_tokens_per_second": 188.679,
        "response": "Hello",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "elapsed_ms": 12.5,
        "generated_at": "2026-10-22T00:00:00+00:00",
        "local_only": True,
    }
    monkeypatch.setattr(engine_api, "generate_local_stream", lambda *_args, **_kwargs: iter([
        {"type": "token", "response": "Hel", "completion_tokens": 1},
        {"type": "complete", "result": completed},
    ]))

    response = _client().post("/api/engine/generate/stream", json={
        "model_path": str(model), "prompt": "hello", "max_new_tokens": 2, "temperature": 0,
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {"type": "token", "response": "Hel", "completion_tokens": 1}
    assert events[1] == {"type": "complete", "result": completed}
    assert events[1]["result"]["sampling"] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": None,
    }


def test_native_engine_pull_endpoint_starts_model_studio_job(monkeypatch) -> None:
    observed: dict[str, object] = {}
    job = {"job_id": "native-job-123", "status": "queued", "operation": "pull_native"}

    def fake_start_job(config):
        observed.update(config)
        return job

    monkeypatch.setattr(engine_api.model_studio, "start_job", fake_start_job)
    response = _client().post("/api/engine/pull", json={"repo_id": "Qwen/Qwen2.5-0.5B-Instruct"})

    assert response.status_code == 202, response.text
    assert response.json() == job
    assert observed == {"operation": "pull_native", "repo_id": "Qwen/Qwen2.5-0.5B-Instruct"}


def test_native_engine_pull_endpoint_rejects_invalid_repository_id() -> None:
    response = _client().post("/api/engine/pull", json={"repo_id": "../escape"})
    assert response.status_code == 422



def test_openai_models_lists_only_discovered_native_models(monkeypatch) -> None:
    monkeypatch.setattr(openai_compat, "discover_local_engine_assets", lambda: {
        "models": [
            {"name": "tiny-llama", "path": "/trusted/models/tiny", "model_type": "llama"},
            {"name": "unsupported-model", "path": "/trusted/models/other", "model_type": "gemma"},
            {"name": "qwen", "path": "/trusted/models/qwen", "model_type": "qwen2"},
        ],
    })
    response = _client().get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [item["id"] for item in payload["data"]] == ["tiny-llama", "qwen"]
    assert all(item["owned_by"] == "axiom" for item in payload["data"])


def test_openai_chat_completion_calls_axiom_native_stream_with_chat_messages(monkeypatch) -> None:
    monkeypatch.setattr(openai_compat, "discover_local_engine_assets", lambda: {
        "models": [{"name": "tiny-llama", "path": "/trusted/models/tiny", "model_type": "llama"}],
    })
    observed: dict[str, object] = {}

    def fake_stream(**kwargs):
        observed.update(kwargs)
        yield {"type": "complete", "result": {
            "response": "Hello there.",
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "stop_reason": "eos_token",
        }}

    monkeypatch.setattr(openai_compat, "generate_local_stream", fake_stream)
    response = _client().post("/v1/chat/completions", json={
        "model": "tiny-llama",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello."},
        ],
        "temperature": 0,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "Hello there."}
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert observed["model_path"] == "/trusted/models/tiny"
    assert observed["chat_messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello."},
    ]
    assert observed["include_token_events"] is False


def test_openai_chat_stream_formats_cumulative_tokens_as_sse_deltas(monkeypatch) -> None:
    monkeypatch.setattr(openai_compat, "discover_local_engine_assets", lambda: {
        "models": [{"name": "tiny-llama", "path": "/trusted/models/tiny", "model_type": "llama"}],
    })

    def fake_stream(**_kwargs):
        yield {"type": "token", "response": "H", "completion_tokens": 1}
        yield {"type": "telemetry", "metrics": {"stage": "decode"}}
        yield {"type": "token", "response": "Hello", "completion_tokens": 2}
        yield {"type": "complete", "result": {"response": "Hello", "stop_reason": "max_new_tokens"}}

    monkeypatch.setattr(openai_compat, "generate_local_stream", fake_stream)
    response = _client().post("/v1/chat/completions", json={
        "model": "tiny-llama", "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content": "H"' in response.text
    assert '"content": "ello"' in response.text
    assert '"finish_reason": "length"' in response.text
    assert "data: [DONE]" in response.text


def test_openai_chat_rejects_unknown_native_model_and_unsupported_message_role(monkeypatch) -> None:
    monkeypatch.setattr(openai_compat, "discover_local_engine_assets", lambda: {"models": []})
    missing = _client().post("/v1/chat/completions", json={
        "model": "not-installed", "messages": [{"role": "user", "content": "hello"}],
    })
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
    invalid = _client().post("/v1/chat/completions", json={
        "model": "not-installed", "messages": [{"role": "tool", "content": "hello"}],
    })
    assert invalid.status_code == 422


def test_openai_bridge_lists_ollama_models_with_explicit_namespace(monkeypatch) -> None:
    monkeypatch.delenv("AXIOM_OPENAI_API_TOKEN", raising=False)
    monkeypatch.setattr(openai_compat, "_native_models", lambda: [
        {"name": "tiny-native", "path": "/trusted/tiny-native", "model_type": "llama"}
    ])
    monkeypatch.setattr(openai_compat, "_ollama_model_names", lambda: ["qwen2.5:0.5b", "deepseek-r1:8b"])

    response = _client().get("/v1/models")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [
        "tiny-native", "ollama/qwen2.5:0.5b", "ollama/deepseek-r1:8b"
    ]
    assert response.json()["data"][0]["owned_by"] == "axiom"
    assert response.json()["data"][1]["owned_by"] == "ollama"


def test_openai_bridge_routes_namespaced_ollama_chat_with_sampling_options(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_OPENAI_API_TOKEN", "bridge-token")
    monkeypatch.setattr(openai_compat, "_ollama_model_names", lambda: ["qwen2.5:0.5b"])
    captured = {}

    def fake_chat(model, messages, **kwargs):
        captured.update({"model": model, "messages": messages, **kwargs})
        return {
            "model": model,
            "response": "Local Ollama answer",
            "prompt_eval_count": 9,
            "eval_count": 4,
            "done_reason": "stop",
        }

    monkeypatch.setattr(openai_compat, "run_local_chat", fake_chat)
    response = _client().post("/v1/chat/completions", headers={
        "Authorization": "Bearer bridge-token",
    }, json={
        "model": "ollama/qwen2.5:0.5b",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 30,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 4,
        "repetition_penalty": 1.2,
        "seed": 42,
        "stop": ["END"],
    })

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ollama/qwen2.5:0.5b"
    assert body["choices"][0]["message"]["content"] == "Local Ollama answer"
    assert body["usage"] == {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    assert captured == {
        "model": "qwen2.5:0.5b",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
        "num_predict": 30,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 4,
        "repetition_penalty": 1.2,
        "seed": 42,
        "stop_sequences": ["END"],
    }
    assert "bridge-token" not in response.text


def test_openai_bridge_rejects_unknown_namespaced_ollama_model(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_OPENAI_API_TOKEN", "bridge-token")
    monkeypatch.setattr(openai_compat, "_ollama_model_names", lambda: ["qwen2.5:0.5b"])
    response = _client().post("/v1/chat/completions", headers={
        "Authorization": "Bearer bridge-token",
    }, json={
        "model": "ollama/other-model:latest",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"



def test_engine_fabric_endpoint_returns_backend_matrix(monkeypatch) -> None:
    expected = {
        "schema_version": 1,
        "profile": "low-memory",
        "backends": [{"id": "llama-cpp", "status": "ready"}],
        "recommendation": {"backend_id": "llama-cpp", "status": "ready"},
    }
    monkeypatch.setattr(engine_api, "discover_engine_backends", lambda profile: {**expected, "profile": profile})
    response = _client().get("/api/engine/backends?profile=low-memory")
    assert response.status_code == 200
    assert response.json() == expected


def test_engine_fabric_endpoint_rejects_unknown_profile() -> None:
    response = _client().get("/api/engine/backends?profile=remote-cloud")
    assert response.status_code == 422


def test_engine_fabric_chat_endpoint_dispatches_validated_local_request(monkeypatch) -> None:
    observed = {}
    expected = {
        "schema_version": 1, "backend_id": "ollama", "backend_name": "Ollama",
        "model": "tiny-local", "response": "Hello from a local backend.", "done": True,
        "prompt_eval_count": 5, "eval_count": 4, "total_duration_ms": 12.0,
        "generated_at": "2026-10-24T00:00:00+00:00", "execution": "local_loopback",
        "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
    }

    def fake_dispatch(backend_id, model, messages, *, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty, seed, stop_sequences):
        observed.update(backend_id=backend_id, model=model, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, stop_sequences=stop_sequences)
        return expected

    monkeypatch.setattr(engine_api, "run_engine_backend_chat", fake_dispatch)
    response = _client().post("/api/engine/backends/chat", json={
        "backend_id": "ollama", "model": "tiny-local",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_new_tokens": 64, "temperature": 0.2, "top_p": 0.8,
        "top_k": 40, "min_p": 0.15, "repetition_penalty": 1.15, "seed": 123, "stop_sequences": ["<END>"],
    })
    assert response.status_code == 200
    assert response.json() == expected
    assert observed["backend_id"] == "ollama"
    assert observed["max_new_tokens"] == 64
    assert observed["messages"] == [{"role": "user", "content": "Say hello."}]
    assert observed["top_p"] == 0.8 and observed["top_k"] == 40
    assert observed["min_p"] == 0.15
    assert observed["repetition_penalty"] == 1.15 and observed["seed"] == 123
    assert observed["stop_sequences"] == ["<END>"]


def test_engine_fabric_chat_rejects_external_and_unknown_targets() -> None:
    invalid = _client().post("/api/engine/backends/chat", json={
        "backend_id": "openai", "model": "tiny-local",
        "messages": [{"role": "user", "content": "Do not route externally."}],
    })
    assert invalid.status_code == 422


def test_engine_fabric_benchmark_endpoint_returns_metadata_only_result(monkeypatch) -> None:
    expected = {
        "schema_version": 1,
        "backend_id": "ollama",
        "backend_name": "Ollama",
        "model": "tiny-local",
        "configuration": {"iterations": 2, "measured_requests": 2},
        "measurements": {"latency_ms": {"median": 15.0, "p95": 19.0}, "requests": []},
        "privacy": {"prompt_returned": False, "responses_returned": False, "results_persisted": False},
        "scope_note": "Measured local calls only.",
    }
    captured = {}
    def fake_benchmark(*args, **kwargs):
        captured.update(kwargs)
        return expected
    monkeypatch.setattr(engine_api, "run_engine_backend_benchmark", fake_benchmark)
    response = _client().post("/api/engine/backends/benchmark", json={
        "backend_id": "llama-cpp", "model": "tiny-local", "prompt": "Measure this.",
        "max_new_tokens": 32, "warmup_runs": 0, "iterations": 2, "temperature": 0.2,
        "top_p": 0.7, "top_k": 32, "min_p": 0.12, "repetition_penalty": 1.2, "seed": 41, "stop_sequences": ["HALT"],
    })
    assert response.status_code == 200
    assert response.json() == expected
    assert captured["top_p"] == 0.7 and captured["top_k"] == 32
    assert captured["min_p"] == 0.12
    assert captured["repetition_penalty"] == 1.2 and captured["seed"] == 41
    assert captured["stop_sequences"] == ["HALT"]


def test_engine_fabric_benchmark_endpoint_rejects_large_run_budget() -> None:
    response = _client().post("/api/engine/backends/benchmark", json={
        "backend_id": "llama-cpp", "model": "tiny-local", "prompt": "Measure this.",
        "max_new_tokens": 2048, "warmup_runs": 1, "iterations": 10,
    })
    assert response.status_code == 422


def test_engine_fabric_auto_chat_endpoint_routes_with_requested_profile(monkeypatch) -> None:
    captured = {}
    expected = {
        "schema_version": 1, "backend_id": "ollama", "backend_name": "Ollama",
        "model": "tiny-local", "response": "Routed locally.", "done": True,
        "prompt_eval_count": 3, "eval_count": 4, "total_duration_ms": 20.0,
        "generated_at": "2026-10-24T00:00:00+00:00", "execution": "local_loopback",
        "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
        "routing": {
            "strategy": "profile_order_exact_model_id", "profile": "low-memory",
            "requested_model": "tiny-local", "fallback_used": False,
            "attempts": [{"backend_id": "ollama", "backend_name": "Ollama", "outcome": "selected"}],
            "max_attempts": 3, "scope_note": "Fixed loopback only.",
        },
    }
    def fake_route(profile, model, messages, *, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty, seed, stop_sequences):
        captured.update(profile=profile, model=model, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, stop_sequences=stop_sequences)
        return expected

    monkeypatch.setattr(engine_api, "run_engine_fabric_chat", fake_route)
    response = _client().post("/api/engine/backends/auto-chat", json={
        "profile": "low-memory", "model": "tiny-local",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_new_tokens": 64, "temperature": 0.2, "top_p": 0.75,
        "top_k": 8, "min_p": 0.08, "repetition_penalty": 1.1, "seed": 7, "stop_sequences": ["STOP"],
    })
    assert response.status_code == 200
    assert response.json() == expected
    assert captured == {
        "profile": "low-memory", "model": "tiny-local",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_new_tokens": 64, "temperature": 0.2,
        "top_p": 0.75, "top_k": 8, "min_p": 0.08, "repetition_penalty": 1.1,
        "seed": 7, "stop_sequences": ["STOP"],
    }


def test_engine_fabric_auto_chat_endpoint_rejects_unknown_profiles() -> None:
    response = _client().post("/api/engine/backends/auto-chat", json={
        "profile": "cloud", "model": "tiny-local",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert response.status_code == 422


def test_engine_fabric_native_backend_chat_endpoint_dispatches_in_process(monkeypatch):
    from axiom.core import llm_client

    observed = {}

    def fake_native(provider, model, messages, **kwargs):
        observed.update(provider=provider, model=model, messages=messages, **kwargs)
        return {
            "response": "native answer", "prompt_tokens": 3, "completion_tokens": 2,
            "device": "cpu", "precision": "fp32", "generated_at": "2026-10-24T00:00:00+00:00",
        }

    monkeypatch.setattr(llm_client, "chat", fake_native)
    response = _client().post("/api/engine/backends/chat", json={
        "backend_id": "axiom-native",
        "model": "qwen-tiny",
        "messages": [{"role": "user", "content": "Say hello."}],
        "top_p": 0.8, "top_k": 32, "min_p": 0.07, "repetition_penalty": 1.1,
        "seed": 9, "stop_sequences": ["END"],
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["backend_id"] == "axiom-native"
    assert result["execution"] == "in_process_native"
    assert result["response"] == "native answer"
    assert observed["provider"] == "axiom-local"
    assert observed["top_p"] == 0.8 and observed["top_k"] == 32
    assert observed["min_p"] == 0.07
    assert observed["repetition_penalty"] == 1.1 and observed["seed"] == 9
    assert observed["stop_sequences"] == ("END",)

def test_engine_fabric_auto_chat_does_not_expose_internal_exception_details(
    monkeypatch,
) -> None:
    secret_detail = "internal failure at /private/runner/credential-store"

    def fail_route(*_args, **_kwargs):
        raise ValueError(secret_detail)

    monkeypatch.setattr(engine_api, "run_engine_fabric_chat", fail_route)
    response = _client().post("/api/engine/backends/auto-chat", json={
        "profile": "balanced",
        "model": "tiny-local",
        "messages": [{"role": "user", "content": "Hello"}],
    })

    assert response.status_code == 422
    assert response.json()["detail"] == "The routing request is invalid or unsupported."
    assert secret_detail not in response.text
    assert "/private/runner" not in response.text

