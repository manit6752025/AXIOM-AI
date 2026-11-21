from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import hardware_snapshots as snapshots_api
from axiom.core import hardware_snapshots as snapshots_core
from axiom.core.hardware_snapshots import HardwareSnapshotStore


def test_hardware_snapshot_api_requires_opt_in_and_exposes_explicit_lifecycle(tmp_path, monkeypatch) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    monkeypatch.setattr(snapshots_api, "HardwareSnapshotStore", lambda: store)
    monkeypatch.setattr(snapshots_core, "detect_hardware", lambda: SimpleNamespace(
        os_name="Linux", architecture="x86_64", cpu_cores=8, ram_gb=32.0,
        gpu_name="Test GPU", vram_gb=12.0, cuda_available=True,
        hostname="DO-NOT-RETURN",
    ))
    app = FastAPI()
    app.include_router(snapshots_api.router)
    client = TestClient(app)

    initial = client.get("/api/workspace/hardware-snapshots")
    assert initial.status_code == 200
    assert initial.json()["settings"]["enabled"] is False
    assert initial.json()["total"] == 0

    blocked = client.post("/api/workspace/hardware-snapshots")
    assert blocked.status_code == 409
    assert "disabled" in blocked.json()["detail"]

    configured = client.put("/api/workspace/hardware-snapshots/settings", json={
        "enabled": True, "max_snapshots": 5, "max_age_days": 30,
    })
    assert configured.status_code == 200
    assert configured.json()["settings"]["enabled"] is True

    captured = client.post("/api/workspace/hardware-snapshots")
    assert captured.status_code == 201
    assert captured.json()["history"]["total"] == 1
    assert "DO-NOT-RETURN" not in captured.text
    assert captured.json()["history"]["privacy"]["automatic_sampling"] is False

    cleared = client.delete("/api/workspace/hardware-snapshots")
    assert cleared.status_code == 200
    assert cleared.json()["total"] == 0
    assert cleared.json()["settings"]["enabled"] is True


def test_hardware_snapshot_api_rejects_out_of_range_retention_and_extra_fields(tmp_path, monkeypatch) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    monkeypatch.setattr(snapshots_api, "HardwareSnapshotStore", lambda: store)
    app = FastAPI()
    app.include_router(snapshots_api.router)
    client = TestClient(app)

    invalid = client.put("/api/workspace/hardware-snapshots/settings", json={
        "enabled": True, "max_snapshots": 101, "max_age_days": 30,
    })
    assert invalid.status_code == 422
    extra = client.put("/api/workspace/hardware-snapshots/settings", json={
        "enabled": True, "max_snapshots": 5, "max_age_days": 30, "host": "not-accepted",
    })
    assert extra.status_code == 422
