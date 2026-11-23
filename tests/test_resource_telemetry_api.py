from __future__ import annotations

from datetime import UTC, datetime
from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import engine as engine_api
from axiom.core.resource_telemetry import ResourceTelemetryStore


def telemetry_sample():
    return {
        "schema_version": 1,
        "sampled_at": datetime.now(UTC).isoformat(),
        "sample_type": "point_in_time",
        "cpu": {"logical_cores": 8, "system_percent": 10.0, "process_percent": 1.0},
        "memory": {"total_bytes": 1024, "available_bytes": 512, "used_bytes": 512, "percent": 50.0},
        "process": {"pid": 77, "rss_bytes": 128, "peak_rss_bytes": 256, "threads": 4},
        "devices": [],
        "loaded_runtime": {
            "loaded": False, "model": "DO-NOT-RETURN", "architecture": None, "device": None,
            "memory_mode": None, "precision": None, "last_generation": None, "device_allocator": None,
        },
        "disks": [], "privacy_scope": {"credentials_included": False},
    }


def test_resource_telemetry_api_has_explicit_opt_in_record_export_and_clear_lifecycle(tmp_path, monkeypatch) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    monkeypatch.setattr(engine_api, "ResourceTelemetryStore", lambda: store)
    telemetry_calls = []
    monkeypatch.setattr(engine_api, "runtime_telemetry", lambda: telemetry_calls.append(True) or telemetry_sample())
    app = FastAPI()
    app.include_router(engine_api.router)
    client = TestClient(app)

    initial = client.get("/api/engine/telemetry-history")
    assert initial.status_code == 200
    assert initial.json()["settings"]["enabled"] is False
    assert initial.json()["samples"] == []
    blocked = client.post("/api/engine/telemetry-history")
    assert blocked.status_code == 409
    assert telemetry_calls == []

    enabled = client.put("/api/engine/telemetry-history/settings", json={
        "enabled": True, "max_samples": 20, "max_age_days": 14,
    })
    assert enabled.status_code == 200
    recorded = client.post("/api/engine/telemetry-history")
    assert recorded.status_code == 201
    payload = recorded.json()
    assert payload["sample"]["metrics"]["process"]["rss_bytes"] == 128
    assert payload["history"]["total"] == 1
    assert "DO-NOT-RETURN" not in recorded.text
    assert '"pid"' not in recorded.text
    assert telemetry_calls == [True]

    cleared = client.delete("/api/engine/telemetry-history")
    assert cleared.status_code == 200
    assert cleared.json()["total"] == 0
    assert cleared.json()["settings"]["enabled"] is True


def test_resource_telemetry_api_rejects_invalid_retention_settings_and_extra_fields(tmp_path, monkeypatch) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    monkeypatch.setattr(engine_api, "ResourceTelemetryStore", lambda: store)
    app = FastAPI()
    app.include_router(engine_api.router)
    client = TestClient(app)
    invalid = client.put("/api/engine/telemetry-history/settings", json={
        "enabled": True, "max_samples": 501, "max_age_days": 14,
    })
    assert invalid.status_code == 422
    extra = client.put("/api/engine/telemetry-history/settings", json={
        "enabled": True, "max_samples": 20, "max_age_days": 14, "hostname": "not-accepted",
    })
    assert extra.status_code == 422

    non_boolean = client.put("/api/engine/telemetry-history/settings", json={
        "enabled": "true", "max_samples": 20, "max_age_days": 14,
    })
    assert non_boolean.status_code == 422
    invalid_limit = client.get("/api/engine/telemetry-history?limit=501")
    assert invalid_limit.status_code == 422



def test_resource_telemetry_history_accepts_valid_http_query_limits(tmp_path, monkeypatch) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    monkeypatch.setattr(engine_api, "ResourceTelemetryStore", lambda: store)
    app = FastAPI()
    app.include_router(engine_api.router)
    client = TestClient(app)

    default_page = client.get("/api/engine/telemetry-history")
    explicit_page = client.get("/api/engine/telemetry-history?limit=500")
    small_page = client.get("/api/engine/telemetry-history?limit=20")

    assert default_page.status_code == 200, default_page.text
    assert explicit_page.status_code == 200, explicit_page.text
    assert small_page.status_code == 200, small_page.text
    assert default_page.json()["limit"] == 500
    assert explicit_page.json()["limit"] == 500
    assert small_page.json()["limit"] == 20
    assert default_page.json()["returned"] == 0
    assert default_page.json()["total"] == 0

    below_minimum = client.get("/api/engine/telemetry-history?limit=0")
    above_maximum = client.get("/api/engine/telemetry-history?limit=501")
    malformed = client.get("/api/engine/telemetry-history?limit=not-a-number")
    assert below_minimum.status_code == 422
    assert above_maximum.status_code == 422
    assert malformed.status_code == 422
