from __future__ import annotations

import json
import os

import pytest

from axiom.core.engine_fabric_profiles import EngineFabricProfileError, EngineFabricProfileStore


def test_engine_fabric_profiles_persist_only_fixed_connector_metadata(tmp_path) -> None:
    store = EngineFabricProfileStore(tmp_path)
    first = store.save(name="Office llama.cpp", backend_id="llama-cpp", model="qwen2.5:7b-instruct")
    updated = store.save(name="office LLAMA.CPP", backend_id="llama-cpp", model="qwen2.5:14b-instruct")

    assert first["id"] == updated["id"]
    assert updated["model"] == "qwen2.5:14b-instruct"
    report = store.list()
    assert report["total"] == 1
    assert report["credentials_persisted"] is False
    assert report["custom_endpoints_persisted"] is False
    raw = store.path.read_text(encoding="utf-8")
    assert "api_key" not in raw
    assert "endpoint" not in raw
    assert "prompt" not in raw
    assert set(json.loads(raw)) == {"schema_version", "profiles"}
    assert set(json.loads(raw)["profiles"][0]) == {
        "id", "name", "backend_id", "model", "created_at", "updated_at",
    }
    if os.name != "nt":
        assert (store.path.stat().st_mode & 0o777) & 0o077 == 0


def test_engine_fabric_profiles_validate_backend_model_and_forbid_extra_fields(tmp_path) -> None:
    store = EngineFabricProfileStore(tmp_path)
    with pytest.raises(EngineFabricProfileError, match="fixed local Engine Fabric backend"):
        store.save(name="Remote endpoint", backend_id="https://example.com/v1", model="model")
    with pytest.raises(EngineFabricProfileError, match="valid exact model names"):
        store.save(name="Invalid model", backend_id="ollama", model="../outside")
    profile = store.save(name="Good profile", backend_id="ollama", model="qwen2.5:0.5b")
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["profiles"][0]["endpoint"] = "http://169.254.169.254/"
    store.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EngineFabricProfileError, match="secret-bearing fields"):
        store.list()
    assert profile["backend_id"] == "ollama"


def test_engine_fabric_profiles_support_explicit_deletion_and_reject_invalid_ids(tmp_path) -> None:
    store = EngineFabricProfileStore(tmp_path)
    profile = store.save(name="Delete this", backend_id="axiom-native", model="tiny-llama")
    assert store.get(profile["id"]) == profile
    assert store.delete(profile["id"]) == profile
    assert store.get(profile["id"]) is None
    assert store.delete(profile["id"]) is None
    with pytest.raises(EngineFabricProfileError, match="Invalid Engine Fabric profile ID"):
        store.delete("../outside")


def test_engine_fabric_profiles_enforce_workspace_limit(tmp_path) -> None:
    store = EngineFabricProfileStore(tmp_path)
    for index in range(25):
        store.save(name=f"Profile {index:02d}", backend_id="ollama", model=f"model-{index}")
    with pytest.raises(EngineFabricProfileError, match="limit of 25"):
        store.save(name="Profile overflow", backend_id="ollama", model="overflow")
