"""Opt-in, workspace-local history of privacy-bounded hardware snapshots."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from axiom.core.hardware import detect_hardware
from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir
from axiom.version import __version__

SCHEMA_VERSION = 1
MAX_STORE_BYTES = 1024 * 1024
MAX_SNAPSHOTS = 100
MAX_RETENTION_DAYS = 3650
DEFAULT_MAX_SNAPSHOTS = 20
DEFAULT_MAX_AGE_DAYS = 90
_LOCK_NAME = ".hardware-snapshots.lock"
_SAFE_TEXT = re.compile(r"[^\x20-\x7e]+")


class HardwareSnapshotError(ValueError):
    """An invalid or unsafe hardware snapshot operation."""


class HardwareSnapshotOptInRequired(HardwareSnapshotError):
    """Snapshot capture is disabled until the workspace owner opts in."""


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 40:
        raise HardwareSnapshotError("A hardware snapshot timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HardwareSnapshotError("A hardware snapshot timestamp is invalid.") from None
    if parsed.tzinfo is None:
        raise HardwareSnapshotError("Hardware snapshot timestamps must include a timezone.")
    return parsed.astimezone(UTC)


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise HardwareSnapshotError(f"Hardware {label} must be text.")
    cleaned = "".join(character for character in value if character.isprintable()).strip()
    if not cleaned or len(cleaned) > maximum:
        raise HardwareSnapshotError(f"Hardware {label} is missing or exceeds its size limit.")
    return cleaned


def _optional_gpu_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HardwareSnapshotError("Hardware GPU name must be text or null.")
    cleaned = "".join(character for character in value if character.isprintable()).strip()
    if not cleaned:
        return None
    return cleaned[:128]


def _memory_bytes(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise HardwareSnapshotError(f"Hardware {label} must be a finite non-negative number or null.")
    amount = float(value)
    if amount < 0 or amount > 1_000_000:
        raise HardwareSnapshotError(f"Hardware {label} exceeds its supported range.")
    if amount == 0:
        return None
    return int(round(amount * (1024 ** 3)))


def _normalize_hardware(value: Any) -> dict[str, Any]:
    """Copy only approved hardware fields; never store arbitrary detector metadata."""
    if isinstance(value, dict) and set(value) == {
        "os_name", "architecture", "cpu_cores", "ram_bytes",
        "gpu_name", "vram_bytes", "cuda_available",
    }:
        os_name = _text(value.get("os_name"), "OS name", 64)
        architecture = _text(value.get("architecture"), "architecture", 64)
        cpu_cores = value.get("cpu_cores")
        ram_bytes = value.get("ram_bytes")
        vram_bytes = value.get("vram_bytes")
        gpu_name = _optional_gpu_name(value.get("gpu_name"))
        cuda_available = value.get("cuda_available")
        if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or not 1 <= cpu_cores <= 4096:
            raise HardwareSnapshotError("Hardware CPU core count must be an integer from 1 through 4096.")
        for label, amount in (("RAM", ram_bytes), ("VRAM", vram_bytes)):
            if amount is not None and (
                isinstance(amount, bool) or not isinstance(amount, int) or not 1 <= amount <= 1024 ** 5
            ):
                raise HardwareSnapshotError(f"Hardware {label} byte count is invalid.")
        if type(cuda_available) is not bool:
            raise HardwareSnapshotError("Hardware CUDA availability must be a boolean.")
        return {
            "os_name": os_name, "architecture": architecture,
            "cpu_cores": cpu_cores, "ram_bytes": ram_bytes,
            "gpu_name": gpu_name, "vram_bytes": vram_bytes,
            "cuda_available": cuda_available,
        }

    def get(name: str) -> Any:
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    os_name = _text(get("os_name"), "OS name", 64)
    architecture = _text(get("architecture"), "architecture", 64)
    cpu_cores = get("cpu_cores")
    if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or not 1 <= cpu_cores <= 4096:
        raise HardwareSnapshotError("Hardware CPU core count must be an integer from 1 through 4096.")
    cuda_available = get("cuda_available")
    if type(cuda_available) is not bool:
        raise HardwareSnapshotError("Hardware CUDA availability must be a boolean.")
    gpu_name = _optional_gpu_name(get("gpu_name"))
    vram_bytes = _memory_bytes(get("vram_gb"), "VRAM")
    if gpu_name is None:
        vram_bytes = None
        cuda_available = False
    return {
        "os_name": os_name,
        "architecture": architecture,
        "cpu_cores": cpu_cores,
        "ram_bytes": _memory_bytes(get("ram_gb"), "RAM"),
        "gpu_name": gpu_name,
        "vram_bytes": vram_bytes,
        "cuda_available": cuda_available,
    }


def _default_settings() -> dict[str, Any]:
    return {
        "enabled": False,
        "max_snapshots": DEFAULT_MAX_SNAPSHOTS,
        "max_age_days": DEFAULT_MAX_AGE_DAYS,
    }


def _normalize_settings(value: Any) -> dict[str, Any]:
    expected = {"enabled", "max_snapshots", "max_age_days"}
    if not isinstance(value, dict) or set(value) != expected:
        raise HardwareSnapshotError("Hardware snapshot retention settings are malformed.")
    enabled = value.get("enabled")
    max_snapshots = value.get("max_snapshots")
    max_age_days = value.get("max_age_days")
    if type(enabled) is not bool:
        raise HardwareSnapshotError("Hardware snapshot opt-in must be a boolean.")
    if isinstance(max_snapshots, bool) or not isinstance(max_snapshots, int) or not 1 <= max_snapshots <= MAX_SNAPSHOTS:
        raise HardwareSnapshotError(f"Hardware snapshot retention must keep 1–{MAX_SNAPSHOTS} entries.")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or not 1 <= max_age_days <= MAX_RETENTION_DAYS:
        raise HardwareSnapshotError(f"Hardware snapshot age retention must be 1–{MAX_RETENTION_DAYS} days.")
    return {
        "enabled": enabled,
        "max_snapshots": max_snapshots,
        "max_age_days": max_age_days,
    }


class HardwareSnapshotStore:
    """Store explicit hardware captures without recording host identity or workload data."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else workspace_metadata_dir()
        self.path = self.root / "hardware-snapshots.json"
        self.lock_path = self.root / _LOCK_NAME

    def _check_paths(self, *, create: bool = False) -> None:
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise HardwareSnapshotError("The hardware snapshot store must not use symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise HardwareSnapshotError("The workspace metadata location is not a directory.")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise HardwareSnapshotError("The hardware snapshot store path changed unsafely.")
        if self.root.exists() and os.name != "nt":

    def _read(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._check_paths()
        if not self.path.exists():
            return _default_settings(), []
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise HardwareSnapshotError("The hardware snapshot store is not a bounded regular file.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except HardwareSnapshotError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise HardwareSnapshotError("The hardware snapshot store could not be read safely.") from None
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "settings", "snapshots"}:
            raise HardwareSnapshotError("The hardware snapshot document is malformed.")
        version = raw.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise HardwareSnapshotError("Unsupported hardware snapshot schema version.")
        settings = _normalize_settings(raw.get("settings"))
        snapshots = raw.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) > MAX_SNAPSHOTS:
            raise HardwareSnapshotError("The hardware snapshot history exceeds its entry limit.")
        normalized: list[dict[str, Any]] = []
        ids: set[str] = set()
        allowed_hardware = {
            "os_name", "architecture", "cpu_cores", "ram_bytes",
            "gpu_name", "vram_bytes", "cuda_available",
        }
        for item in snapshots:
            if not isinstance(item, dict) or set(item) != {
                "id", "captured_at", "fingerprint", "source", "software_version", "hardware",
            }:
                raise HardwareSnapshotError("A hardware snapshot contains unsupported fields.")
            identity = item.get("id")
            if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity) or identity in ids:
                raise HardwareSnapshotError("The hardware snapshot history contains an invalid or duplicate ID.")
            captured_at = _timestamp(item.get("captured_at")).isoformat()
            fingerprint = item.get("fingerprint")
            if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                raise HardwareSnapshotError("The hardware snapshot fingerprint is invalid.")
            if item.get("source") != "explicit_capture":
                raise HardwareSnapshotError("The hardware snapshot source is invalid.")
            software_version = _text(item.get("software_version"), "software version", 64)
            hardware = item.get("hardware")
            if not isinstance(hardware, dict) or set(hardware) != allowed_hardware:
                raise HardwareSnapshotError("A hardware snapshot contains unexpected hardware fields.")
            safe_hardware = _normalize_hardware(hardware)
            canonical = json.dumps(safe_hardware, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != fingerprint:
                raise HardwareSnapshotError("A hardware snapshot fingerprint does not match its content.")
            ids.add(identity)
            normalized.append({
                "id": identity,
                "captured_at": captured_at,
                "fingerprint": fingerprint,
                "source": "explicit_capture",
                "software_version": software_version,
                "hardware": safe_hardware,
            })
        normalized.sort(key=lambda record: record["captured_at"], reverse=True)
        return settings, normalized

    @staticmethod
    def _retained(
        snapshots: list[dict[str, Any]],
        settings: dict[str, Any],
        now: datetime,
    ) -> list[dict[str, Any]]:
        cutoff = now - timedelta(days=settings["max_age_days"])
        retained = [
            item for item in snapshots
            if _timestamp(item["captured_at"]) >= cutoff
        ]
        retained.sort(key=lambda item: item["captured_at"], reverse=True)
        return retained[:settings["max_snapshots"]]

    def _write(self, settings: dict[str, Any], snapshots: list[dict[str, Any]]) -> None:
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "settings": settings, "snapshots": snapshots},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise HardwareSnapshotError("The hardware snapshot store exceeds its 1 MiB size limit.")
        atomic_write_text(self.path, payload)
        if os.name != "nt":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                raise HardwareSnapshotError("Could not secure hardware snapshot file permissions.") from None

    def _report(
        self,
        settings: dict[str, Any],
        snapshots: list[dict[str, Any]],
        *,
        limit: int,
    ) -> dict[str, Any]:
        retained = self._retained(snapshots, settings, _now())
        return {
            "schema_version": SCHEMA_VERSION,
            "settings": settings,
            "total": len(retained),
            "snapshots": retained[:limit],
            "privacy": {
                "host_identity_included": False,
                "network_identifiers_included": False,
                "filesystem_paths_included": False,
                "credentials_included": False,
                "model_or_dataset_content_included": False,
                "automatic_sampling": False,
            },
            "scope_note": (
                "History contains only explicit, local hardware captures. Capture is disabled by default. "
                "Retention is bounded by both entry count and age; no hostnames, network identifiers, file paths, "
                "environment values, credentials, model files, or workload content are stored."
            ),
        }

    def list(self, *, limit: int = MAX_SNAPSHOTS) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SNAPSHOTS:
            raise HardwareSnapshotError(f"Snapshot list limit must be 1–{MAX_SNAPSHOTS}.")
        settings, snapshots = self._read()
        return self._report(settings, snapshots, limit=limit)

    def configure(
        self,
        *,
        enabled: bool,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> dict[str, Any]:
        settings = _normalize_settings({
            "enabled": enabled,
            "max_snapshots": max_snapshots,
            "max_age_days": max_age_days,
        })
        self._check_paths(create=True)
        try:
            with file_lock(self.lock_path):
                self._check_paths(create=True)
                current, snapshots = self._read()
                # Disabling stops new capture; previously captured records are retained
                # until their configured age/count limits or an explicit future clear.
                retained = self._retained(snapshots, settings, _now())
                self._write(settings, retained)
        except HardwareSnapshotError:
            raise
        except (OSError, TimeoutError):
            raise HardwareSnapshotError("Hardware snapshot settings could not be saved safely.") from None
        return self._report(settings, retained, limit=MAX_SNAPSHOTS)

    def capture(self, hardware: Any | None = None) -> dict[str, Any]:
        # Fail before creating the workspace metadata directory or lock file
        # when the owner has not explicitly opted in.
        self._check_paths()
        settings, _snapshots = self._read()
        if not settings["enabled"]:
            raise HardwareSnapshotOptInRequired(
                "Hardware snapshot history is disabled. Enable it explicitly before capturing a snapshot."
            )
        self._check_paths(create=True)
        try:
            with file_lock(self.lock_path):
                self._check_paths(create=True)
                settings, snapshots = self._read()
                if not settings["enabled"]:
                    raise HardwareSnapshotOptInRequired(
                        "Hardware snapshot history is disabled. Enable it explicitly before capturing a snapshot."
                    )
                observed = detect_hardware() if hardware is None else hardware
                safe_hardware = _normalize_hardware(observed)
                canonical = json.dumps(safe_hardware, sort_keys=True, separators=(",", ":"), allow_nan=False)
                now = _now()
                record = {
                    "id": uuid.uuid4().hex,
                    "captured_at": now.isoformat(),
                    "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "source": "explicit_capture",
                    "software_version": __version__[:64],
                    "hardware": safe_hardware,
                }
                retained = self._retained([record, *snapshots], settings, now)
                self._write(settings, retained)
        except HardwareSnapshotError:
            raise
        except (OSError, TimeoutError):
            raise HardwareSnapshotError("The hardware snapshot could not be saved safely.") from None
        return {"snapshot": record, "history": self._report(settings, retained, limit=MAX_SNAPSHOTS)}


    def clear(self) -> dict[str, Any]:
        """Delete every retained hardware snapshot while keeping opt-in settings."""
        self._check_paths(create=True)
        try:
            with file_lock(self.lock_path):
                self._check_paths(create=True)
                settings, _snapshots = self._read()
                self._write(settings, [])
        except HardwareSnapshotError:
            raise
        except (OSError, TimeoutError):
            raise HardwareSnapshotError("The hardware snapshot history could not be cleared safely.") from None
        return self._report(settings, [], limit=MAX_SNAPSHOTS)
