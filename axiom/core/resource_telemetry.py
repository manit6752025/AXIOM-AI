"""Opt-in, privacy-bounded persistent resource telemetry history."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from axiom.core.storage import atomic_write_text, file_lock
from axiom.path_safety import open_regular_file
from axiom.core.workspace_context import workspace_metadata_dir

SCHEMA_VERSION = 1
MAX_STORE_BYTES = 4 * 1024 * 1024
MAX_SAMPLES = 500
MAX_RETENTION_DAYS = 3650
DEFAULT_MAX_SAMPLES = 120
DEFAULT_MAX_AGE_DAYS = 30
_LOCK_NAME = ".resource-telemetry.lock"
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ResourceTelemetryError(ValueError):
    """An invalid, malformed, or unsafe resource telemetry operation."""


class ResourceTelemetryOptInRequired(ResourceTelemetryError):
    """Persistent resource sampling is disabled until explicitly enabled."""


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 40:
        raise ResourceTelemetryError("A resource telemetry timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ResourceTelemetryError("A resource telemetry timestamp is invalid.") from None
    if parsed.tzinfo is None:
        raise ResourceTelemetryError("Resource telemetry timestamps must include a timezone.")
    return parsed.astimezone(UTC)


def _optional_number(value: Any, label: str, maximum: float, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ResourceTelemetryError(f"Resource telemetry {label} must be a finite number or null.")
    number = float(value)
    if number < 0 or number > maximum:
        raise ResourceTelemetryError(f"Resource telemetry {label} is outside its supported range.")
    return int(number) if integer else number


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResourceTelemetryError(f"Resource telemetry {label} must be text or null.")
    cleaned = "".join(character for character in value if character.isprintable()).strip()
    if not cleaned:
        return None
    if len(cleaned) > maximum:
        raise ResourceTelemetryError(f"Resource telemetry {label} exceeds its size limit.")
    return cleaned


def _normalize_allocator(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ResourceTelemetryError("Resource telemetry allocator metrics must be an object or null.")
    backend = _optional_text(value.get("backend"), "allocator backend", 48)
    flags: dict[str, bool] = {}
    for name in ("peak_reset_supported", "device_allocator_metrics_supported"):
        flag = value.get(name)
        if type(flag) is not bool:
            raise ResourceTelemetryError(f"Resource telemetry allocator {name} must be boolean.")
        flags[name] = flag
    values = {
        name: _optional_number(value.get(name), name, 2**63 - 1, integer=True)
        for name in (
            "current_allocated_bytes", "current_reserved_bytes", "peak_allocated_bytes",
            "peak_reserved_bytes", "peak_increment_bytes", "process_rss_bytes",
            "process_rss_delta_bytes", "process_peak_rss_bytes", "process_peak_rss_delta_bytes",
        )
    }
    return {"backend": backend, **values, **flags}


def _normalize_metrics(value: Any) -> dict[str, Any]:
    """Copy approved telemetry values and discard identity, paths and workload details."""
    if not isinstance(value, dict):
        raise ResourceTelemetryError("A resource telemetry sample must be an object.")
    sampled_at = _timestamp(value.get("sampled_at")).isoformat()
    if value.get("sample_type") != "point_in_time":
        raise ResourceTelemetryError("Only point-in-time resource telemetry samples are supported.")

    cpu = value.get("cpu")
    memory = value.get("memory")
    process = value.get("process")
    loaded = value.get("loaded_runtime")
    devices = value.get("devices")
    if not all(isinstance(item, dict) for item in (cpu, memory, process, loaded)):
        raise ResourceTelemetryError("A resource telemetry sample is missing a required metrics object.")
    if not isinstance(devices, list) or len(devices) > 16:
        raise ResourceTelemetryError("Resource telemetry device metrics exceed the supported count.")

    logical_cores = cpu.get("logical_cores")
    if logical_cores is not None and (
        isinstance(logical_cores, bool) or not isinstance(logical_cores, int) or not 1 <= logical_cores <= 4096
    ):
        raise ResourceTelemetryError("Resource telemetry logical CPU count must be 1–4096 or null.")
    system_cpu = _optional_number(cpu.get("system_percent"), "system CPU percent", 100)
    process_cpu = _optional_number(cpu.get("process_percent"), "process CPU percent", 409600)
    memory_percent = _optional_number(memory.get("percent"), "memory percent", 100)

    safe_devices: list[dict[str, Any]] = []
    for item in devices:
        if not isinstance(item, dict) or type(item.get("available")) is not bool:
            raise ResourceTelemetryError("A resource telemetry device record is malformed.")
        safe_devices.append({
            "available": item["available"],
            "memory_free_gib": _optional_number(item.get("memory_free_gib"), "device free memory", 1_000_000),
            "memory_total_gib": _optional_number(item.get("memory_total_gib"), "device total memory", 1_000_000),
        })

    loaded_flag = loaded.get("loaded")
    if type(loaded_flag) is not bool:
        raise ResourceTelemetryError("Resource telemetry loaded-runtime status must be boolean.")
    memory_mode = loaded.get("memory_mode")
    if memory_mode not in (None, "full", "layer_streaming", "disk_streaming"):
        raise ResourceTelemetryError("Resource telemetry memory mode is unsupported.")
    return {
        "sampled_at": sampled_at,
        "sample_type": "point_in_time",
        "cpu": {
            "logical_cores": logical_cores,
            "system_percent": system_cpu,
            "process_percent": process_cpu,
        },
        "memory": {
            "total_bytes": _optional_number(memory.get("total_bytes"), "total RAM bytes", 2**63 - 1, integer=True),
            "available_bytes": _optional_number(memory.get("available_bytes"), "available RAM bytes", 2**63 - 1, integer=True),
            "used_bytes": _optional_number(memory.get("used_bytes"), "used RAM bytes", 2**63 - 1, integer=True),
            "percent": memory_percent,
        },
        "process": {
            "rss_bytes": _optional_number(process.get("rss_bytes"), "process RSS bytes", 2**63 - 1, integer=True),
            "peak_rss_bytes": _optional_number(process.get("peak_rss_bytes"), "process peak RSS bytes", 2**63 - 1, integer=True),
            "threads": _optional_number(process.get("threads"), "process threads", 100_000, integer=True),
        },
        "devices": safe_devices,
        "loaded_runtime": {
            "loaded": loaded_flag,
            "device": _optional_text(loaded.get("device"), "runtime device", 64),
            "memory_mode": memory_mode,
            "precision": _optional_text(loaded.get("precision"), "runtime precision", 32),
            "device_allocator": _normalize_allocator(loaded.get("device_allocator")),
        },
    }


def _default_settings() -> dict[str, Any]:
    return {"enabled": False, "max_samples": DEFAULT_MAX_SAMPLES, "max_age_days": DEFAULT_MAX_AGE_DAYS}


def _normalize_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"enabled", "max_samples", "max_age_days"}:
        raise ResourceTelemetryError("Resource telemetry retention settings are malformed.")
    enabled = value.get("enabled")
    max_samples = value.get("max_samples")
    max_age_days = value.get("max_age_days")
    if type(enabled) is not bool:
        raise ResourceTelemetryError("Resource telemetry opt-in must be boolean.")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= MAX_SAMPLES:
        raise ResourceTelemetryError(f"Resource telemetry retention must keep 1–{MAX_SAMPLES} samples.")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or not 1 <= max_age_days <= MAX_RETENTION_DAYS:
        raise ResourceTelemetryError(f"Resource telemetry age retention must be 1–{MAX_RETENTION_DAYS} days.")
    return {"enabled": enabled, "max_samples": max_samples, "max_age_days": max_age_days}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Resource telemetry JSON contains duplicate object keys.")
        result[key] = value
    return result


class ResourceTelemetryStore:
    """Workspace-local history containing only allowlisted resource metrics."""

    def __init__(self, root: Path | None = None) -> None:
        selected = Path(root).expanduser() if root is not None else Path(workspace_metadata_dir())
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        self.root = Path(os.path.abspath(selected))
        self.path = self.root / "resource-telemetry.json"
        self.lock_path = self.root / _LOCK_NAME

    @staticmethod
    def _supports_directory_descriptors() -> bool:
        return (
            os.name == "posix"
            and hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
            and os.open in getattr(os, "supports_dir_fd", set())
            and os.mkdir in getattr(os, "supports_dir_fd", set())
            and os.unlink in getattr(os, "supports_dir_fd", set())
            and os.replace in getattr(os, "supports_dir_fd", set())
        )

    def _check_paths(self, *, create: bool = False) -> None:
        current = Path(self.root.anchor)
        for component in self.root.parts[1:]:
            current = current / component
            if current.is_symlink():
                raise ResourceTelemetryError("The resource telemetry metadata path must not contain symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise ResourceTelemetryError("The workspace metadata location is not a directory.")
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise ResourceTelemetryError("The resource telemetry store and lock must not be symbolic links.")
        if self.path.exists():
            try:
                info = self.path.lstat()
            except OSError:
                raise ResourceTelemetryError("The resource telemetry store could not be inspected safely.") from None
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise ResourceTelemetryError("The resource telemetry store is not a bounded regular file.")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise ResourceTelemetryError("The resource telemetry store path changed unsafely.")

    def _open_directory(self, *, create: bool) -> int | None:
        if not self._supports_directory_descriptors():
            self._check_paths(create=create)
            return None if not self.root.exists() else -1
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(self.root.anchor, flags)
        try:
            for component in self.root.parts[1:]:
                try:
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        return None
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode):
                raise ResourceTelemetryError("The workspace metadata location is not a directory.")
            if info.st_mode & 0o077:
                os.fchmod(descriptor, 0o700)
            result = descriptor
            descriptor = -1
            return result
        except ResourceTelemetryError:
            raise
        except OSError:
            raise ResourceTelemetryError("The workspace metadata directory contains an unsafe or unavailable path.") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @contextmanager
    def _locked_directory(self, *, create: bool) -> Iterator[int | None]:
        """Lock history while anchoring POSIX file operations to a no-follow directory."""
        if not self._supports_directory_descriptors():
            self._check_paths(create=create)
            if not self.root.exists():
                yield None
                return
            try:
                with file_lock(self.lock_path):
                    yield None
            except (OSError, TimeoutError):
                raise ResourceTelemetryError("The resource telemetry history lock could not be acquired safely.") from None
            return

        directory_fd = self._open_directory(create=create)
        if directory_fd is None:
            yield None
            return
        lock_fd: int | None = None
        try:
            lock_fd = os.open(
                self.lock_path.name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
            lock_info = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_info.st_mode):
                raise ResourceTelemetryError("The resource telemetry lock must be a regular file.")
            if lock_info.st_mode & 0o077:
                os.fchmod(lock_fd, 0o600)
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield directory_fd
        except ResourceTelemetryError:
            raise
        except OSError:
            raise ResourceTelemetryError("The resource telemetry history lock could not be acquired safely.") from None
        finally:
            if lock_fd is not None:
                try:
                    import fcntl
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(directory_fd)

    @staticmethod
    def _retained(samples: list[dict[str, Any]], settings: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
        cutoff = now - timedelta(days=settings["max_age_days"])
        retained = [sample for sample in samples if _timestamp(sample["sampled_at"]) >= cutoff]
        retained.sort(key=lambda sample: sample["sampled_at"], reverse=True)
        return retained[:settings["max_samples"]]

    def _read(self, directory_fd: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        owns_directory = directory_fd is None
        selected_fd = self._open_directory(create=False) if owns_directory else directory_fd
        if selected_fd is None:
            return _default_settings(), []
        try:
            try:
                if selected_fd >= 0:
                    descriptor = os.open(
                        self.path.name,
                        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=selected_fd,
                    )
                    with os.fdopen(descriptor, "rb") as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                            raise ResourceTelemetryError("The resource telemetry store is not a bounded regular file.")
                        raw_bytes = stream.read(MAX_STORE_BYTES + 1)
                else:
                    self._check_paths()
                    with open_regular_file(self.path) as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                            raise ResourceTelemetryError("The resource telemetry store is not a bounded regular file.")
                        raw_bytes = stream.read(MAX_STORE_BYTES + 1)
                if len(raw_bytes) > MAX_STORE_BYTES:
                    raise ResourceTelemetryError("The resource telemetry store exceeds its 4 MiB size limit.")
                raw = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
            except FileNotFoundError:
                return _default_settings(), []
            except ResourceTelemetryError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
                raise ResourceTelemetryError("The resource telemetry store could not be read safely.") from None
        finally:
            if owns_directory and selected_fd >= 0:
                os.close(selected_fd)

        if not isinstance(raw, dict) or set(raw) != {"schema_version", "settings", "samples"}:
            raise ResourceTelemetryError("The resource telemetry document is malformed.")
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != SCHEMA_VERSION:
            raise ResourceTelemetryError("Unsupported resource telemetry schema version.")
        settings = _normalize_settings(raw.get("settings"))
        samples = raw.get("samples")
        if not isinstance(samples, list) or len(samples) > MAX_SAMPLES:
            raise ResourceTelemetryError("The resource telemetry history exceeds its sample limit.")
        normalized: list[dict[str, Any]] = []
        identities: set[str] = set()
        for item in samples:
            if not isinstance(item, dict) or set(item) != {"id", "sampled_at", "fingerprint", "metrics"}:
                raise ResourceTelemetryError("A resource telemetry record contains unsupported fields.")
            identity = item.get("id")
            fingerprint = item.get("fingerprint")
            if not isinstance(identity, str) or not _ID_PATTERN.fullmatch(identity) or identity in identities:
                raise ResourceTelemetryError("The resource telemetry history contains an invalid or duplicate ID.")
            if not isinstance(fingerprint, str) or not _HASH_PATTERN.fullmatch(fingerprint):
                raise ResourceTelemetryError("A resource telemetry fingerprint is invalid.")
            metrics = _normalize_metrics(item.get("metrics"))
            sampled_at = _timestamp(item.get("sampled_at")).isoformat()
            if sampled_at != metrics["sampled_at"]:
                raise ResourceTelemetryError("A resource telemetry record timestamp does not match its sample.")
            canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != fingerprint:
                raise ResourceTelemetryError("A resource telemetry fingerprint does not match its content.")
            identities.add(identity)
            normalized.append({
                "id": identity,
                "sampled_at": sampled_at,
                "fingerprint": fingerprint,
                "metrics": metrics,
            })
        return settings, self._retained(normalized, settings, datetime.now(UTC))

    def _write(
        self,
        settings: dict[str, Any],
        samples: list[dict[str, Any]],
        directory_fd: int | None,
    ) -> None:
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "settings": settings, "samples": samples},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_STORE_BYTES:
            raise ResourceTelemetryError("The resource telemetry store exceeds its 4 MiB size limit.")
        if directory_fd is None or directory_fd < 0:
            self._check_paths(create=True)
            atomic_write_text(self.path, payload)
            if os.name != "nt":
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    raise ResourceTelemetryError("Could not secure resource telemetry file permissions.") from None
            return

        temporary_name = f".resource-telemetry.{secrets.token_hex(16)}.tmp"
        descriptor: int | None = None
        created = False
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
            created = True
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary_name,
                self.path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            created = False
            os.fsync(directory_fd)
        except OSError:
            raise ResourceTelemetryError("The resource telemetry store could not be written atomically and safely.") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if created:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except OSError:
                    pass

    def _report(self, settings: dict[str, Any], samples: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
        retained = self._retained(samples, settings, datetime.now(UTC))
        ordered = list(reversed(retained))[-limit:]
        return {
            "schema_version": SCHEMA_VERSION,
            "settings": settings,
            "total": len(retained),
            "samples": ordered,
            "privacy": {
                "process_ids_included": False,
                "model_names_or_paths_included": False,
                "prompt_or_generated_content_included": False,
                "environment_values_included": False,
                "credentials_included": False,
                "network_identifiers_included": False,
                "automatic_sampling": False,
            },
        }

    def list(self, *, limit: int = MAX_SAMPLES) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SAMPLES:
            raise ResourceTelemetryError(f"Resource telemetry list limit must be 1–{MAX_SAMPLES}.")
        settings, samples = self._read()
        return self._report(settings, samples, limit=limit)

    def configure(
        self,
        *,
        enabled: bool,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> dict[str, Any]:
        settings = _normalize_settings({
            "enabled": enabled,
            "max_samples": max_samples,
            "max_age_days": max_age_days,
        })
        try:
            with self._locked_directory(create=True) as directory_fd:
                current, samples = self._read(directory_fd)
                del current
                retained = self._retained(samples, settings, datetime.now(UTC))
                self._write(settings, retained, directory_fd)
        except ResourceTelemetryError:
            raise
        except (OSError, TimeoutError):
            raise ResourceTelemetryError("Resource telemetry retention settings could not be saved safely.") from None
        return self._report(settings, retained, limit=MAX_SAMPLES)

    def record(self, sample: Any) -> dict[str, Any]:
        # Do not create workspace state until opt-in has been saved explicitly.
        settings, _samples = self._read()
        if not settings["enabled"]:
            raise ResourceTelemetryOptInRequired(
                "Persistent resource history is disabled. Enable it explicitly before recording samples."
            )
        try:
            with self._locked_directory(create=True) as directory_fd:
                settings, samples = self._read(directory_fd)
                if not settings["enabled"]:
                    raise ResourceTelemetryOptInRequired(
                        "Persistent resource history is disabled. Enable it explicitly before recording samples."
                    )
                metrics = _normalize_metrics(sample)
                canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False)
                record = {
                    "id": uuid.uuid4().hex,
                    "sampled_at": metrics["sampled_at"],
                    "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "metrics": metrics,
                }
                retained = self._retained([*samples, record], settings, datetime.now(UTC))
                self._write(settings, retained, directory_fd)
        except ResourceTelemetryError:
            raise
        except (OSError, TimeoutError):
            raise ResourceTelemetryError("The resource telemetry sample could not be saved safely.") from None
        return {"sample": record, "history": self._report(settings, retained, limit=MAX_SAMPLES)}

    def clear(self) -> dict[str, Any]:
        """Remove samples while retaining the explicit opt-in and retention settings."""
        self._check_paths()
        if not self.path.exists():
            return self._report(_default_settings(), [], limit=MAX_SAMPLES)
        try:
            with self._locked_directory(create=False) as directory_fd:
                if directory_fd is None:
                    return self._report(_default_settings(), [], limit=MAX_SAMPLES)
                settings, _samples = self._read(directory_fd)
                self._write(settings, [], directory_fd)
        except ResourceTelemetryError:
            raise
        except (OSError, TimeoutError):
            raise ResourceTelemetryError("Resource telemetry history could not be cleared safely.") from None
        return self._report(settings, [], limit=MAX_SAMPLES)
