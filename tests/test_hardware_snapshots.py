from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import pytest

from axiom.core import hardware_snapshots as snapshots
from axiom.core.hardware_snapshots import (
    HardwareSnapshotError,
    HardwareSnapshotOptInRequired,
    HardwareSnapshotStore,
)


def hardware(**extra):
    values = {
        "os_name": "Linux",
        "architecture": "x86_64",
        "cpu_cores": 16,
        "ram_gb": 64.0,
        "gpu_name": "Test accelerator",
        "vram_gb": 24.0,
        "cuda_available": True,
        "hostname": "NEVER-PERSIST-HOSTNAME",
        "model_path": "/private/models/weights.gguf",
        "api_key": "NEVER-PERSIST-SECRET",
    }
    values.update(extra)
    return SimpleNamespace(**values)


def test_history_is_disabled_by_default_and_reads_never_probe_hardware(tmp_path, monkeypatch) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    monkeypatch.setattr(snapshots, "detect_hardware", lambda: pytest.fail("history reads must not probe hardware"))
    report = store.list()
    assert report["settings"] == {"enabled": False, "max_snapshots": 20, "max_age_days": 90}
    assert report["total"] == 0
    assert report["privacy"]["automatic_sampling"] is False
    assert not store.root.exists()
    with pytest.raises(HardwareSnapshotOptInRequired):
        store.capture()
    assert not store.root.exists()


def test_explicit_capture_persists_only_allowlisted_hardware_and_private_file(tmp_path) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    store.configure(enabled=True)
    captured = store.capture(hardware())
    record = captured["snapshot"]
    assert record["source"] == "explicit_capture"
    assert record["hardware"] == {
        "os_name": "Linux",
        "architecture": "x86_64",
        "cpu_cores": 16,
        "ram_bytes": 64 * (1024 ** 3),
        "gpu_name": "Test accelerator",
        "vram_bytes": 24 * (1024 ** 3),
        "cuda_available": True,
    }
    serialized = store.path.read_text(encoding="utf-8")
    assert "NEVER-PERSIST" not in serialized
    assert "/private/models" not in serialized
    assert "hostname" not in serialized
    assert "api_key" not in serialized
    assert len(record["fingerprint"]) == 64
    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


def test_retention_is_bounded_by_count_and_age_and_disable_preserves_records(tmp_path) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    store.configure(enabled=True, max_snapshots=1, max_age_days=7)
    first = store.capture(hardware())["snapshot"]
    second = store.capture(hardware(cpu_cores=32))["snapshot"]
    assert store.list()["total"] == 1
    assert store.list()["snapshots"][0]["id"] == second["id"]
    disabled = store.configure(enabled=False, max_snapshots=1, max_age_days=7)
    assert disabled["settings"]["enabled"] is False
    assert disabled["total"] == 1
    with pytest.raises(HardwareSnapshotOptInRequired):
        store.capture(hardware())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["snapshots"][0]["captured_at"] = "2000-01-01T00:00:00+00:00"
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    pruned = store.configure(enabled=False, max_snapshots=1, max_age_days=7)
    assert pruned["total"] == 0
    assert first["id"] != second["id"]


def test_clear_removes_records_and_preserves_opt_in_setting(tmp_path) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    store.configure(enabled=True)
    store.capture(hardware())
    result = store.clear()
    assert result["total"] == 0
    assert result["settings"]["enabled"] is True
    assert store.list()["snapshots"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_snapshots": 0},
        {"max_snapshots": 101},
        {"max_age_days": 0},
        {"max_age_days": 3651},
    ],
)
def test_retention_bounds_reject_invalid_values(tmp_path, kwargs) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    with pytest.raises(HardwareSnapshotError):
        store.configure(enabled=True, **kwargs)


def test_store_rejects_tampered_and_unknown_snapshot_fields(tmp_path) -> None:
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    store.configure(enabled=True)
    store.capture(hardware())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["snapshots"][0]["hardware"]["hostname"] = "unexpected"
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HardwareSnapshotError, match="unexpected hardware fields"):
        store.list()


@pytest.mark.skipif(os.name == "nt", reason="Directory symlink privileges vary on Windows CI.")
def test_store_rejects_symlinked_metadata_root(tmp_path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(HardwareSnapshotError, match="symbolic links"):
        HardwareSnapshotStore(link).list()
