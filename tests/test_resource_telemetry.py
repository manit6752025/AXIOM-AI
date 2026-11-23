from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta

import pytest

from axiom.core.resource_telemetry import (
    ResourceTelemetryError,
    ResourceTelemetryOptInRequired,
    ResourceTelemetryStore,
)


def sample(**overrides):
    value = {
        "schema_version": 1,
        "sampled_at": datetime.now(UTC).isoformat(),
        "sample_type": "point_in_time",
        "cpu": {"logical_cores": 16, "system_percent": 21.5, "process_percent": 12.0},
        "memory": {
            "total_bytes": 64 * 1024**3, "available_bytes": 24 * 1024**3,
            "used_bytes": 40 * 1024**3, "percent": 62.5,
        },
        "process": {"pid": 12345, "rss_bytes": 1024, "peak_rss_bytes": 2048, "threads": 8},
        "devices": [{
            "id": "private-device-id", "name": "Private GPU name", "available": True,
            "memory_free_gib": 12.0, "memory_total_gib": 24.0,
        }],
        "loaded_runtime": {
            "loaded": True, "model": "private-model-name", "architecture": "PrivateArchitecture",
            "device": "cuda:0", "memory_mode": "layer_streaming", "precision": "fp16",
            "last_generation": {"prompt": "NEVER-PERSIST-PROMPT", "completion": "NEVER-PERSIST-OUTPUT"},
            "device_allocator": {
                "backend": "cuda", "current_allocated_bytes": 1024, "current_reserved_bytes": 2048,
                "peak_allocated_bytes": 4096, "peak_reserved_bytes": 8192, "peak_increment_bytes": 1024,
                "process_rss_bytes": 1024, "process_rss_delta_bytes": 0,
                "process_peak_rss_bytes": 2048, "process_peak_rss_delta_bytes": 0,
                "peak_reset_supported": True, "device_allocator_metrics_supported": True,
            },
        },
        "disks": [{"label": "private path", "free_bytes": 100}],
        "privacy_scope": {"credentials_included": False},
    }
    value.update(overrides)
    return value


def test_history_is_disabled_by_default_and_does_not_create_workspace_files(tmp_path) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    report = store.list()
    assert report["settings"] == {"enabled": False, "max_samples": 120, "max_age_days": 30}
    assert report["samples"] == []
    assert report["privacy"]["automatic_sampling"] is False
    with pytest.raises(ResourceTelemetryOptInRequired):
        store.record(sample())
    assert not store.root.exists()


def test_record_persists_allowlisted_metrics_only_and_uses_private_file_modes(tmp_path) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    store.configure(enabled=True)
    result = store.record(sample())
    record = result["sample"]
    assert record["metrics"]["cpu"]["system_percent"] == 21.5
    assert "pid" not in record["metrics"]["process"]
    assert "model" not in record["metrics"]["loaded_runtime"]
    assert "last_generation" not in record["metrics"]["loaded_runtime"]
    assert "id" not in record["metrics"]["devices"][0]
    assert "name" not in record["metrics"]["devices"][0]
    raw = store.path.read_text(encoding="utf-8")
    for secret in ("NEVER-PERSIST", "private-model-name", "PrivateArchitecture", "private-device-id", "private path", '"pid"'):
        assert secret not in raw
    assert result["history"]["total"] == 1
    assert len(record["fingerprint"]) == 64
    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


def test_sample_retention_is_bounded_by_count_and_age_and_disable_stops_new_records(tmp_path) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    store.configure(enabled=True, max_samples=1, max_age_days=7)
    first = store.record(sample())["sample"]
    second = store.record(sample(cpu={"logical_cores": 16, "system_percent": 35.0, "process_percent": 2.0}))["sample"]
    listed = store.list()
    assert listed["total"] == 1
    assert listed["samples"][0]["id"] == second["id"]
    assert first["id"] != second["id"]
    disabled = store.configure(enabled=False, max_samples=1, max_age_days=7)
    assert disabled["settings"]["enabled"] is False
    with pytest.raises(ResourceTelemetryOptInRequired):
        store.record(sample())


def test_age_retention_prunes_old_records_and_clear_preserves_settings(tmp_path) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    store.configure(enabled=True, max_samples=10, max_age_days=7)
    store.record(sample())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["samples"][0]["sampled_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    payload["samples"][0]["metrics"]["sampled_at"] = payload["samples"][0]["sampled_at"]
    import hashlib
    canonical = json.dumps(payload["samples"][0]["metrics"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["samples"][0]["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.list()["total"] == 0
    cleared = store.clear()
    assert cleared["settings"]["enabled"] is True
    assert cleared["total"] == 0


@pytest.mark.parametrize("settings", [
    {"enabled": True, "max_samples": 0, "max_age_days": 7},
    {"enabled": True, "max_samples": 501, "max_age_days": 7},
    {"enabled": True, "max_samples": 10, "max_age_days": 0},
    {"enabled": True, "max_samples": 10, "max_age_days": 3651},
])
def test_invalid_retention_settings_are_rejected(tmp_path, settings) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    with pytest.raises(ResourceTelemetryError):
        store.configure(**settings)


def test_history_rejects_tampered_fingerprints_and_unsupported_fields(tmp_path) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    store.configure(enabled=True)
    store.record(sample())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["samples"][0]["metrics"]["cpu"]["system_percent"] = 99.0
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResourceTelemetryError, match="fingerprint"):
        store.list()


def test_history_serializes_concurrent_sample_writes(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = ResourceTelemetryStore(tmp_path / ".axiom")
    store.configure(enabled=True, max_samples=120, max_age_days=30)

    def write_sample(index: int):
        return store.record(sample(cpu={
            "logical_cores": 16,
            "system_percent": float(index),
            "process_percent": float(index) / 2,
        }))["sample"]["id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        identifiers = list(executor.map(write_sample, range(24)))

    history = store.list()
    assert len(set(identifiers)) == 24
    assert history["total"] == 24
    assert {record["id"] for record in history["samples"]} == set(identifiers)


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow directory-descriptor support is required.")
def test_history_rejects_symlinked_store_and_lock_files(tmp_path) -> None:
    root = tmp_path / ".axiom"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"private":"sentinel"}', encoding="utf-8")
    store = ResourceTelemetryStore(root)

    store.path.symlink_to(outside)
    with pytest.raises(ResourceTelemetryError):
        store.list()
    store.path.unlink()

    lock_target = tmp_path / "lock-target"
    lock_target.write_text("lock sentinel", encoding="utf-8")
    store.lock_path.symlink_to(lock_target)
    with pytest.raises(ResourceTelemetryError):
        store.configure(enabled=True)
    assert lock_target.read_text(encoding="utf-8") == "lock sentinel"


def test_history_rejects_duplicate_json_object_keys(tmp_path) -> None:
    store = ResourceTelemetryStore(tmp_path / ".axiom")
    store.configure(enabled=True)
    store.path.write_text(
        '{"schema_version":1,"schema_version":1,"settings":{"enabled":true,"max_samples":120,"max_age_days":30},"samples":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ResourceTelemetryError, match="duplicate"):
        store.list()


@pytest.mark.skipif(os.name == "nt", reason="Directory symlink privileges vary on Windows CI.")
def test_history_rejects_symlinked_metadata_root(tmp_path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ResourceTelemetryError, match="symbolic links"):
        ResourceTelemetryStore(link).list()
