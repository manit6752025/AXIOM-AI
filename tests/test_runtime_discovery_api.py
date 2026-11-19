from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import runtime_discovery


def test_runtime_connector_route_returns_passive_inventory(monkeypatch) -> None:
    expected = {
        "schema_version": 1,
        "discovery_mode": "passive_local_discovery",
        "checked_at": "2026-10-20T00:00:00+00:00",
        "connectors": [],
        "scope_note": "Presence is not a connection.",
    }
    monkeypatch.setattr(runtime_discovery, "discover_runtime_connectors", lambda: expected)

    app = FastAPI()
    app.include_router(runtime_discovery.router)
    response = TestClient(app).get("/api/diagnostics/runtime-connectors")

    assert response.status_code == 200
    assert response.json() == expected
