"""Workspace-local, secret-free profiles for fixed AXIOM Engine Fabric adapters."""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.inference import validate_model_name
from axiom.core.resource_ids import is_resource_id, new_resource_id
from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir

SCHEMA_VERSION = 1
MAX_PROFILES = 25
MAX_STORE_BYTES = 64 * 1024
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,47}$")
BACKENDS = frozenset({
    "axiom-native",
    "ollama",
    "llama-cpp",
    "lm-studio",
    "vllm",
    "llama-cpp-python",
})
_LOCK_NAME = ".engine-fabric-profiles.lock"


class EngineFabricProfileError(ValueError):
    """Invalid or unsafe local Engine Fabric profile operation."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class EngineFabricProfileStore:
    """Persist connector identity only; endpoints, credentials and request data are forbidden."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else workspace_metadata_dir()
        self.path = self.root / "engine-fabric-profiles.json"
        self.lock_path = self.root / _LOCK_NAME

    def _check_paths(self, *, create: bool = False) -> None:
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise EngineFabricProfileError("The Engine Fabric profile store must not use symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise EngineFabricProfileError("The workspace metadata path is not a directory.")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise EngineFabricProfileError("The Engine Fabric profile store path changed unsafely.")
        if self.root.exists() and os.name != "nt":
            try:
                os.chmod(self.root, 0o700)
            except OSError as exc:
                raise EngineFabricProfileError("Could not restrict Engine Fabric profile directory permissions.") from exc

    @staticmethod
    def _normalize(*, name: Any, backend_id: Any, model: Any) -> tuple[str, str, str]:
        if not isinstance(name, str):
            raise EngineFabricProfileError("Connector profile name must be text.")
        normalized_name = name.strip()
        if not PROFILE_NAME.fullmatch(normalized_name):
            raise EngineFabricProfileError("Connector profile names must be 1–48 safe characters and start with a letter or number.")
        if not isinstance(backend_id, str) or backend_id not in BACKENDS:
            raise EngineFabricProfileError("Choose a supported fixed local Engine Fabric backend.")
        try:
            normalized_model = validate_model_name(model)
        except (TypeError, ValueError) as exc:
            raise EngineFabricProfileError("Connector model IDs must be valid exact model names, not URLs or paths.") from exc
        return normalized_name, backend_id, normalized_model

    def _read(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.path.exists():
            return []
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise EngineFabricProfileError("The Engine Fabric profile store is not a bounded regular file.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except EngineFabricProfileError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise EngineFabricProfileError("The Engine Fabric profile store could not be read safely.") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "profiles"}:
            raise EngineFabricProfileError("The Engine Fabric profile document is malformed.")
        version, profiles = raw.get("schema_version"), raw.get("profiles")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise EngineFabricProfileError("Unsupported Engine Fabric profile schema version.")
        if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
            raise EngineFabricProfileError("The Engine Fabric profile list is invalid or exceeds its limit.")
        normalized: list[dict[str, Any]] = []
        ids: set[str] = set()
        names: set[str] = set()
        expected = {"id", "name", "backend_id", "model", "created_at", "updated_at"}
        for item in profiles:
            if not isinstance(item, dict) or set(item) != expected:
                # Reject unknown fields, including endpoints, keys and prompt content.
                raise EngineFabricProfileError("An Engine Fabric profile contains unsupported or secret-bearing fields.")
            profile_id = item.get("id")
            if not is_resource_id(profile_id):
                raise EngineFabricProfileError("The Engine Fabric profile store contains an invalid ID.")
            name, backend_id, model = self._normalize(
                name=item.get("name"), backend_id=item.get("backend_id"), model=item.get("model"),
            )
            created_at, updated_at = item.get("created_at"), item.get("updated_at")
            if (
                not isinstance(created_at, str) or not created_at or len(created_at) > 40
                or not isinstance(updated_at, str) or not updated_at or len(updated_at) > 40
            ):
                raise EngineFabricProfileError("The Engine Fabric profile timestamps are invalid.")
            if profile_id in ids or name.casefold() in names:
                raise EngineFabricProfileError("Engine Fabric profile IDs and names must be unique.")
            ids.add(profile_id)
            names.add(name.casefold())
            normalized.append({
                "id": profile_id, "name": name, "backend_id": backend_id, "model": model,
                "created_at": created_at, "updated_at": updated_at,
            })
        return normalized

    def list(self) -> dict[str, Any]:
        profiles = self._read()
        return {
            "schema_version": SCHEMA_VERSION,
            "profiles": profiles,
            "total": len(profiles),
            "max_profiles": MAX_PROFILES,
            "credentials_persisted": False,
            "custom_endpoints_persisted": False,
        }

    def get(self, profile_id: str) -> dict[str, Any] | None:
        if not is_resource_id(profile_id):
            raise EngineFabricProfileError("Invalid Engine Fabric profile ID.")
        return next((item for item in self._read() if item["id"] == profile_id), None)

    def _write(self, profiles: list[dict[str, Any]]) -> None:
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "profiles": profiles},
            ensure_ascii=False, indent=2, allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise EngineFabricProfileError("The Engine Fabric profile store exceeds its 64 KiB limit.")
        atomic_write_text(self.path, payload)
        if os.name != "nt":
            try:
                os.chmod(self.path, 0o600)
            except OSError as exc:
                raise EngineFabricProfileError("Could not secure Engine Fabric profile permissions.") from exc

    def save(self, *, name: str, backend_id: str, model: str) -> dict[str, Any]:
        normalized_name, normalized_backend, normalized_model = self._normalize(
            name=name, backend_id=backend_id, model=model,
        )
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            profiles = self._read()
            existing = next((item for item in profiles if item["name"].casefold() == normalized_name.casefold()), None)
            now = _now()
            if existing is None:
                if len(profiles) >= MAX_PROFILES:
                    raise EngineFabricProfileError(f"The Engine Fabric profile limit of {MAX_PROFILES} has been reached.")
                profile = {
                    "id": new_resource_id(), "name": normalized_name, "backend_id": normalized_backend,
                    "model": normalized_model, "created_at": now, "updated_at": now,
                }
                profiles.insert(0, profile)
            else:
                profile = {
                    **existing, "name": normalized_name, "backend_id": normalized_backend,
                    "model": normalized_model, "updated_at": now,
                }
                profiles = [profile if item["id"] == existing["id"] else item for item in profiles]
            self._write(profiles)
            return dict(profile)

    def delete(self, profile_id: str) -> dict[str, Any] | None:
        if not is_resource_id(profile_id):
            raise EngineFabricProfileError("Invalid Engine Fabric profile ID.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            profiles = self._read()
            removed = next((item for item in profiles if item["id"] == profile_id), None)
            if removed is None:
                return None
            self._write([item for item in profiles if item["id"] != profile_id])
            return dict(removed)
