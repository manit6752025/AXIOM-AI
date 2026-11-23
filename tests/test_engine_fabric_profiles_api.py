from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import engine as engine_api
from axiom.core.engine_fabric_profiles import EngineFabricProfileStore


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(engine_api.router)
    return TestClient(app)


def _report(*, status: str = "ready", models: list[str] | None = None) -> dict:
    return {
        "profile": "balanced",
        "backends": [{
            "id": "llama-cpp",
            "name": "llama.cpp server",
            "status": status,
            "models": models if models is not None else ["qwen2.5:7b-instruct"],
        }],
    }


def test_engine_fabric_profile_api_crud_and_one_shot_connect(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(engine_api, "EngineFabricProfileStore", lambda: EngineFabricProfileStore(tmp_path))
    calls = []
    monkeypatch.setattr(engine_api, "discover_engine_backends", lambda profile: calls.append(profile) or _report())
    client = _client()

    created = client.post("/api/engine/backends/profiles", json={
        "name": "Office local model",
        "backend_id": "llama-cpp",
        "model": "qwen2.5:7b-instruct",
    })
    assert created.status_code == 201, created.text
    profile = created.json()["profile"]
    assert created.json()["credentials_persisted"] is False
    assert created.json()["custom_endpoints_persisted"] is False

    listed = client.get("/api/engine/backends/profiles")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["profiles"][0]["id"] == profile["id"]

    connected = client.post(f"/api/engine/backends/profiles/{profile['id']}/connect", json={})
    assert connected.status_code == 200, connected.text
    result = connected.json()
    assert result["status"] == "verified"
    assert result["persistent_connection"] is False
    assert result["connection_scope"] == "one_shot_local_health_and_model_inventory_check"
    assert result["processes_started"] is False
    assert "endpoint" not in result
    assert calls == ["balanced"]
    raw = EngineFabricProfileStore(tmp_path).path.read_text(encoding="utf-8")
    assert "api_key" not in raw and "endpoint" not in raw and "prompt" not in raw

    deleted = client.delete(f"/api/engine/backends/profiles/{profile['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"]["id"] == profile["id"]
    assert client.get("/api/engine/backends/profiles").json()["total"] == 0


def test_engine_fabric_profile_api_rejects_credentials_and_custom_endpoints(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(engine_api, "EngineFabricProfileStore", lambda: EngineFabricProfileStore(tmp_path))
    client = _client()
    invalids = [
        {"name": "Key profile", "backend_id": "ollama", "model": "tiny", "api_key": "must-never-save"},
        {"name": "Endpoint profile", "backend_id": "ollama", "model": "tiny", "endpoint": "http://127.0.0.1:9999"},
    ]
    for payload in invalids:
        response = client.post("/api/engine/backends/profiles", json=payload)
        assert response.status_code == 422
    assert not (tmp_path / "engine-fabric-profiles.json").exists()


def test_engine_fabric_profile_connect_requires_ready_backend_and_exact_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(engine_api, "EngineFabricProfileStore", lambda: EngineFabricProfileStore(tmp_path))
    store = EngineFabricProfileStore(tmp_path)
    profile = store.save(name="Office profile", backend_id="llama-cpp", model="qwen2.5:7b-instruct")
    monkeypatch.setattr(engine_api, "discover_engine_backends", lambda _profile: _report(status="setup_required"))
    client = _client()
    offline = client.post(f"/api/engine/backends/profiles/{profile['id']}/connect", json={})
    assert offline.status_code == 409

    monkeypatch.setattr(engine_api, "discover_engine_backends", lambda _profile: _report(models=["another-model"]))
    missing = client.post(f"/api/engine/backends/profiles/{profile['id']}/connect", json={})
    assert missing.status_code == 409
