from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import axiom.api.engine as engine_api
import axiom.core.engine as engine_core
from axiom.core.serving_profiles import (
    ServingProfileError,
    ServingProfileStore,
    generate_from_profile,
    normalize_profile_config,
    readiness_probe,
    release_profile_request,
    reserve_profile_request,
    validate_serving_prompt,
)


def _config(root, **overrides):
    values = {
        "name": "Local Qwen",
        "model_path": str(root / "models" / "qwen"),
        "adapter_path": None,
        "device_preference": "auto",
        "memory_mode": "auto",
        "precision": "auto",
        "context_length": 4096,
        "max_new_tokens": 256,
        "max_prompt_chars": 12000,
        "max_concurrent_requests": 1,
        "max_queued_requests": 0,
        "request_timeout_seconds": 300,
    }
    values.update(overrides)
    return values


def _client():
    app = FastAPI()
    app.include_router(engine_api.router)
    return TestClient(app)


def _ready_runtime():
    return {
        "ready": True,
        "problems": [],
        "recommended_device": "cpu",
        "devices": [
            {"id": "cpu", "name": "CPU", "available": True},
            {"id": "cuda", "name": "CUDA", "available": False},
        ],
    }


def test_serving_profile_store_is_bounded_unique_and_workspace_local(tmp_path):
    store = ServingProfileStore(tmp_path / ".axiom")
    first = store.save(_config(tmp_path))
    assert first["id"]
    assert first["context_length"] == 4096
    assert store.list()["total"] == 1
    assert store.list()["weights_loaded"] is False

    updated = store.save(_config(tmp_path, name="local qwen", max_new_tokens=128))
    assert updated["id"] == first["id"]
    assert updated["max_new_tokens"] == 128
    assert store.list()["total"] == 1
    assert store.get(first["id"]) == updated
    assert store.delete(first["id"]) == updated
    assert store.list()["total"] == 0


@pytest.mark.parametrize("changes", [
    {"context_length": True},
    {"context_length": 131073},
    {"max_new_tokens": 2049},
    {"max_prompt_chars": 12001},
    {"max_concurrent_requests": 2},
    {"max_queued_requests": 33},
    {"request_timeout_seconds": 4},
    {"device_preference": "remote"},
    {"memory_mode": "unsafe"},
    {"precision": "int4"},
    {"model_path": "models/qwen"},
    {"model_path": "/trusted/models/../outside"},
])
def test_serving_profile_schema_rejects_unbounded_or_untrusted_configuration(tmp_path, changes):
    with pytest.raises(ServingProfileError):
        normalize_profile_config(_config(tmp_path, **changes))


def test_serving_profile_store_rejects_tampered_schema_and_symlink(tmp_path):
    root = tmp_path / ".axiom"
    root.mkdir()
    path = root / "serving-profiles.json"
    path.write_text('{"schema_version":1,"profiles":[{"id":"x","unexpected":"credential"}]}', encoding="utf-8")
    with pytest.raises(ServingProfileError):
        ServingProfileStore(root).list()

    other = tmp_path / "elsewhere.json"
    other.write_text('{"schema_version":1,"profiles":[]}', encoding="utf-8")
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ServingProfileError):
        ServingProfileStore(root).list()




def test_serving_profile_schema_supports_bounded_opt_in_queue_depth(tmp_path):
    profile = ServingProfileStore(tmp_path / ".axiom").save(_config(tmp_path, max_queued_requests=8))
    assert profile["max_concurrent_requests"] == 1
    assert profile["max_queued_requests"] == 8
def test_serving_profile_readiness_reports_runtime_device_and_context_without_loading_weights(tmp_path, monkeypatch):
    profile = ServingProfileStore(tmp_path / ".axiom").save(_config(tmp_path))
    monkeypatch.setattr(engine_core, "runtime_status", _ready_runtime)
    monkeypatch.setattr(engine_core, "inspect_native_model_context", lambda path: {
        "model_type": "qwen2",
        "architecture": "Qwen2ForCausalLM",
        "checkpoint_format": "safetensors",
        "max_position_embeddings": 8192,
        "effective_max_position_embeddings": 8192,
    })

    report = readiness_probe(profile)
    assert report["ready"] is True
    assert report["checks"]["device"]["selected"] == "cpu"
    assert report["checks"]["context"]["available"] is True
    assert report["weights_loaded"] is False
    assert report["service_started"] is False
    assert report["network_used"] is False


def test_serving_profile_readiness_fails_when_device_or_context_is_unavailable(tmp_path, monkeypatch):
    profile = ServingProfileStore(tmp_path / ".axiom").save(
        _config(tmp_path, device_preference="cuda", context_length=8192)
    )
    monkeypatch.setattr(engine_core, "runtime_status", _ready_runtime)
    monkeypatch.setattr(engine_core, "inspect_native_model_context", lambda path: {
        "model_type": "llama", "architecture": "LlamaForCausalLM", "checkpoint_format": "safetensors",
        "max_position_embeddings": 4096, "effective_max_position_embeddings": 4096,
    })

    report = readiness_probe(profile)
    assert report["ready"] is False
    assert report["checks"]["device"]["available"] is False
    assert report["checks"]["context"]["available"] is False
    assert len(report["problems"]) >= 2


def test_serving_profile_readiness_checks_adapter_compatibility_for_selected_device(tmp_path, monkeypatch):
    profile = ServingProfileStore(tmp_path / ".axiom").save(
        _config(tmp_path, adapter_path=str(tmp_path / "models" / "lora"))
    )
    monkeypatch.setattr(engine_core, "runtime_status", _ready_runtime)
    monkeypatch.setattr(engine_core, "inspect_native_model_context", lambda path: {
        "model_type": "qwen2", "architecture": "Qwen2ForCausalLM", "checkpoint_format": "safetensors",
        "max_position_embeddings": 8192, "effective_max_position_embeddings": 8192,
    })
    monkeypatch.setattr(engine_core, "check_local_adapter_compatibility", lambda *args, **kwargs: {
        "structurally_compatible": True,
        "devices": [{"id": "cpu", "compatible": True}, {"id": "cuda", "compatible": False}],
    })

    report = readiness_probe(profile)
    assert report["ready"] is True
    assert report["checks"]["adapter"]["selected_device"] == "cpu"
    assert report["checks"]["adapter"]["compatible"] is True


def test_serving_profile_prompt_preflight_enforces_chars_output_and_exact_context(tmp_path, monkeypatch):
    profile = ServingProfileStore(tmp_path / ".axiom").save(
        _config(tmp_path, context_length=16, max_new_tokens=8, max_prompt_chars=32)
    )
    monkeypatch.setattr(engine_core, "inspect_native_prompt_batch", lambda *args, **kwargs: {
        "available": True,
        "prompts_truncated": 0,
        "prompt_token_counts": {"max": 8},
    })
    budget = validate_serving_prompt(profile, "hello")
    assert budget == {
        "prompt_characters": 5,
        "prompt_tokens": 8,
        "max_new_tokens": 8,
        "requested_context_tokens": 16,
        "context_limit_tokens": 16,
    }
    with pytest.raises(ServingProfileError, match="context"):
        validate_serving_prompt({**profile, "context_length": 15}, "hello")
    with pytest.raises(ServingProfileError, match="Maximum output tokens"):
        validate_serving_prompt(profile, "hello", 9)
    with pytest.raises(ServingProfileError, match="character"):
        validate_serving_prompt(profile, "x" * 33)


def test_serving_profile_prompt_preflight_rejects_unverifiable_tokenizer(tmp_path, monkeypatch):
    profile = ServingProfileStore(tmp_path / ".axiom").save(_config(tmp_path))
    monkeypatch.setattr(engine_core, "inspect_native_prompt_batch", lambda *args, **kwargs: {
        "available": False, "reason_code": "tokenizer_unavailable",
    })
    with pytest.raises(ServingProfileError, match="tokenizer"):
        validate_serving_prompt(profile, "hello")


def test_serving_profile_admission_rejects_parallel_native_requests(tmp_path):
    profile = ServingProfileStore(tmp_path / ".axiom").save(_config(tmp_path))
    other = ServingProfileStore(tmp_path / ".axiom").save(_config(tmp_path, name="Other"))
    reserve_profile_request(profile["id"])
    try:
        with pytest.raises(ServingProfileError, match="does not queue"):
            reserve_profile_request(other["id"])
    finally:
        release_profile_request(profile["id"])
    reserve_profile_request(other["id"])
    release_profile_request(other["id"])


def test_generate_from_profile_applies_saved_runtime_settings_and_prompt_budget(tmp_path, monkeypatch):
    profile = ServingProfileStore(tmp_path / ".axiom").save(
        _config(tmp_path, memory_mode="layer_streaming", precision="bf16", device_preference="cpu")
    )
    observed = {}
    expected = {"response": "local response", "completion_tokens": 2}

    monkeypatch.setattr(
        engine_core, "generate_local",
        lambda model_path, prompt, **kwargs: (observed.update(model_path=model_path, prompt=prompt, **kwargs) or expected),
    )
    result = generate_from_profile(
        profile, "hello",
        prompt_budget={"max_new_tokens": 32, "prompt_tokens": 3},
    )
    assert result == expected
    assert observed["model_path"] == profile["model_path"]
    assert observed["memory_mode"] == "layer_streaming"
    assert observed["precision"] == "bf16"
    assert observed["device_preference"] == "cpu"


def test_serving_profile_api_crud_readiness_and_profile_limited_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path);
    model = tmp_path / "models" / "qwen"
    model.mkdir(parents=True)
    monkeypatch.setattr(engine_api, "_validated_path", lambda value, label: None if value is None else str(Path(value).resolve()))
    monkeypatch.setattr(engine_api, "inspect_native_model_context", lambda path: {
        "model_type": "qwen2", "architecture": "Qwen2ForCausalLM", "checkpoint_format": "safetensors",
        "max_position_embeddings": 8192, "effective_max_position_embeddings": 8192,
    })
    monkeypatch.setattr(engine_api, "readiness_probe", lambda profile: {
        "profile_id": profile["id"], "ready": True, "problems": [],
        "weights_loaded": False, "service_started": False, "network_used": False,
    })
    monkeypatch.setattr(engine_api, "validate_serving_prompt", lambda profile, prompt, max_new_tokens=None: {
        "prompt_characters": len(prompt), "prompt_tokens": 4,
        "max_new_tokens": max_new_tokens or profile["max_new_tokens"],
        "requested_context_tokens": 4 + (max_new_tokens or profile["max_new_tokens"]),
        "context_limit_tokens": profile["context_length"],
    })
    expected = {"engine": "axiom-native-engine", "response": "hello", "completion_tokens": 1}
    admission_calls = []
    releases = []

    def fake_admission(**kwargs):
        admission_calls.append(kwargs)
        return {"queue_position": 2, "waited_ms": 42}

    monkeypatch.setattr(engine_api.NATIVE_PROFILE_ADMISSION, "acquire", fake_admission)
    monkeypatch.setattr(engine_api.NATIVE_PROFILE_ADMISSION, "release", lambda: releases.append(True))

    def fake_generate(profile, prompt, **kwargs):
        return expected

    monkeypatch.setattr(engine_api, "generate_from_profile", fake_generate)
    with _client() as client:
        create = client.post("/api/engine/serving-profiles", json=_config(
            tmp_path, model_path=str(model), name="API profile", max_queued_requests=3,
        ))
        assert create.status_code == 201, create.text
        profile = create.json()["profile"]
        profile_id = profile["id"]
        assert profile["model_path"] == str(model.resolve())
        assert create.json()["scope"]["weights_loaded"] is False

        listing = client.get("/api/engine/serving-profiles")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1
        assert client.get(f"/api/engine/serving-profiles/{profile_id}").status_code == 200

        probe = client.post(f"/api/engine/serving-profiles/{profile_id}/readiness")
        assert probe.status_code == 200
        assert probe.json()["ready"] is True

        result = client.post(f"/api/engine/serving-profiles/{profile_id}/generate", json={
            "prompt": "hello", "max_new_tokens": 32,
        })
        assert result.status_code == 200, result.text
        assert result.json()["profile"]["id"] == profile_id
        assert result.json()["result"] == expected
        assert result.json()["scope"]["network_used"] is False
        assert result.json()["admission"] == {
            "queue_position": 2,
            "queued": True,
            "waited_ms": 42,
            "concurrency_limit": 1,
            "queue_limit": 3,
        }
        assert admission_calls[0]["queue_key"] == profile_id
        assert admission_calls[0]["queue_limit"] == 3
        assert releases == [True]

        excessive = client.post(f"/api/engine/serving-profiles/{profile_id}/generate", json={
            "prompt": "x" * 12001,
        })
        assert excessive.status_code == 422

        removed = client.delete(f"/api/engine/serving-profiles/{profile_id}")
        assert removed.status_code == 200
        assert removed.json()["model_files_deleted"] is False
        assert client.get(f"/api/engine/serving-profiles/{profile_id}").status_code == 404


def test_serving_profile_api_rejects_unavailable_and_invalid_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path);
    monkeypatch.setattr(engine_api, "_validated_path", lambda value, label: value)
    monkeypatch.setattr(engine_api, "inspect_native_model_context", lambda path: {
        "model_type": "llama", "architecture": "LlamaForCausalLM", "checkpoint_format": "safetensors",
        "max_position_embeddings": 1024, "effective_max_position_embeddings": 1024,
    })
    with _client() as client:
        invalid = client.post("/api/engine/serving-profiles", json=_config(tmp_path, context_length=2048))
        assert invalid.status_code == 422
        malformed = client.post("/api/engine/serving-profiles", json=_config(
            tmp_path, max_concurrent_requests=4,
        ))
        assert malformed.status_code == 422


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [
        (engine_api.RequestAdmissionQueueFull, 429),
        (engine_api.RequestAdmissionTimeout, 504),
        (engine_api.RequestAdmissionCancelled, 499),
    ],
)
def test_serving_profile_api_maps_queue_full_timeout_and_cancellation(
    tmp_path, monkeypatch, error_type, expected_status,
):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "models" / "qwen"
    model.mkdir(parents=True)
    monkeypatch.setattr(engine_api, "_validated_path", lambda value, label: None if value is None else str(Path(value).resolve()))
    monkeypatch.setattr(engine_api, "inspect_native_model_context", lambda path: {
        "model_type": "qwen2", "architecture": "Qwen2ForCausalLM", "checkpoint_format": "safetensors",
        "max_position_embeddings": 8192, "effective_max_position_embeddings": 8192,
    })
    monkeypatch.setattr(engine_api, "readiness_probe", lambda profile: {"ready": True, "problems": []})
    monkeypatch.setattr(engine_api, "validate_serving_prompt", lambda *args, **kwargs: {"max_new_tokens": 8})
    with _client() as client:
        profile = client.post("/api/engine/serving-profiles", json=_config(
            tmp_path, model_path=str(model), name="Busy profile", max_queued_requests=2,
        )).json()["profile"]

        def reject(**kwargs):
            assert kwargs["queue_key"] == profile["id"]
            assert kwargs["queue_limit"] == 2
            raise error_type("The native request admission did not succeed.")

        monkeypatch.setattr(engine_api.NATIVE_PROFILE_ADMISSION, "acquire", reject)
        failed = client.post(
            f"/api/engine/serving-profiles/{profile['id']}/generate",
            json={"prompt": "hi"},
        )
        assert failed.status_code == expected_status
        if expected_status == 429:
            assert failed.headers["retry-after"] == "1"
        else:
            assert "retry-after" not in failed.headers



def test_serving_profile_cli_creates_probes_runs_lists_and_deletes(tmp_path, monkeypatch):
    import json
    from typer.testing import CliRunner

    import axiom.cli.engine as engine_cli

    model = tmp_path / "models" / "tiny-qwen"
    model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine_cli, "resolve_local_model_directory", lambda value, label: str(Path(value).resolve()))
    monkeypatch.setattr(engine_cli, "inspect_native_model_context", lambda path: {
        "model_type": "qwen2", "architecture": "Qwen2ForCausalLM", "checkpoint_format": "safetensors",
        "max_position_embeddings": 8192, "effective_max_position_embeddings": 8192,
    })
    monkeypatch.setattr(engine_cli, "readiness_probe", lambda profile: {
        "profile_id": profile["id"], "profile_name": profile["name"], "ready": True,
        "checks": {}, "problems": [], "weights_loaded": False, "service_started": False,
        "network_used": False, "probed_at": "2026-11-21T00:00:00Z", "scope_note": "Metadata only.",
    })
    monkeypatch.setattr(engine_cli, "generate_from_profile", lambda profile, prompt, **kwargs: {
        "response": "profile response", "completion_tokens": 2, "generation_tokens_per_second": 12.5,
        "elapsed_ms": 160, "model": "tiny-qwen",
    })
    runner = CliRunner()
    created = runner.invoke(engine_cli.engine_app, [
        "profile", "create", "Local test", "--model-path", str(model),
        "--device", "cpu", "--context-length", "4096", "--max-new-tokens", "64", "--json",
    ])
    assert created.exit_code == 0, created.output
    profile = json.loads(created.stdout)["profile"]

    listed = runner.invoke(engine_cli.engine_app, ["profile", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)["profiles"][0]["id"] == profile["id"]

    probed = runner.invoke(engine_cli.engine_app, ["profile", "probe", profile["id"], "--json"])
    assert probed.exit_code == 0, probed.output
    assert json.loads(probed.stdout)["ready"] is True

    generated = runner.invoke(engine_cli.engine_app, [
        "profile", "run", profile["id"], "--prompt", "Hello AXIOM", "--json",
    ])
    assert generated.exit_code == 0, generated.output
    generated_report = json.loads(generated.stdout)
    assert generated_report["result"]["response"] == "profile response"
    assert generated_report["admission"] == {"queue_position": 0, "waited_ms": 0}

    removed = runner.invoke(engine_cli.engine_app, [
        "profile", "delete", profile["id"], "--yes",
    ])
    assert removed.exit_code == 0, removed.output
    assert json.loads(removed.stdout)["model_files_deleted"] is False


def test_serving_profile_schema_accepts_bounded_queue_depth(tmp_path):
    profile = ServingProfileStore(tmp_path / ".axiom").save(
        _config(tmp_path, max_queued_requests=8)
    )
    assert profile["max_concurrent_requests"] == 1
    assert profile["max_queued_requests"] == 8
