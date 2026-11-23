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
    monkeypatch.setattr(engine_api, "runtime_telemetry", telemetry_sample)
    app = FastAPI()
    app.include_router(engine_api.router)
    client = TestClient(app)

    initial = client.get("/api/engine/telemetry-history")
    assert initial.status_code == 200
    assert initial.json()["settings"]["enabled"] is False
    assert initial.json()["samples"] == []
    blocked = client.post("/api/engine/telemetry-history")
    assert blocked.status_code == 409

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
