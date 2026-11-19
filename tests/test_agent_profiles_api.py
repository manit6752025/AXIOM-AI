from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import agent_profiles
from axiom.core.agent_profiles import AgentProfileStore


def test_agent_profile_api_create_update_delete(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent_profiles, "AgentProfileStore", lambda: AgentProfileStore(tmp_path))
    app = FastAPI()
    app.include_router(agent_profiles.router)
    client = TestClient(app)

    initial = client.get("/api/agent-profiles")
    assert initial.status_code == 200
    assert initial.json()["profiles"][0]["task_capabilities"] == []
    assert "do not grant" in initial.json()["authorization_note"]

    created = client.post("/api/agent-profiles", json={"name": "Research", "task_capabilities": ["model.inspect"]})
    assert created.status_code == 201
    profile = created.json()["profile"]

    updated = client.put(f"/api/agent-profiles/{profile['id']}", json={"task_capabilities": ["dataset.profile"]})
    assert updated.status_code == 200
    assert updated.json()["profile"]["version"] == 2

    default_delete = client.delete("/api/agent-profiles/default")
    assert default_delete.status_code == 400
    profile_delete = client.delete(f"/api/agent-profiles/{profile['id']}")
    assert profile_delete.status_code == 200


def test_agent_profile_api_rejects_unknown_capability(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent_profiles, "AgentProfileStore", lambda: AgentProfileStore(tmp_path))
    app = FastAPI()
    app.include_router(agent_profiles.router)
    response = TestClient(app).post("/api/agent-profiles", json={"name": "Unsafe", "task_capabilities": ["shell.execute"]})
    assert response.status_code == 422


def test_agent_profile_api_returns_page_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent_profiles, "AgentProfileStore", lambda: AgentProfileStore(tmp_path))
    app = FastAPI()
    app.include_router(agent_profiles.router)
    client = TestClient(app)
    for index in range(3):
        created = client.post("/api/agent-profiles", json={"name": f"Research {index}"})
        assert created.status_code == 201

    first = client.get("/api/agent-profiles?limit=2&offset=0")
    assert first.status_code == 200
    assert len(first.json()["profiles"]) == 2
    assert first.json()["total"] == 4
    assert first.json()["has_more"] is True

    second = client.get("/api/agent-profiles?limit=2&offset=2")
    assert second.status_code == 200
    assert len(second.json()["profiles"]) == 2
    assert second.json()["has_more"] is False

    invalid = client.get("/api/agent-profiles?limit=101")
    assert invalid.status_code == 422
