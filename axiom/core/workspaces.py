"""Bounded, local-only workspace catalog and filesystem isolation boundaries."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.diagnostics import discover_workspace_root
from axiom.core.storage import atomic_write_text, file_lock

_WORKSPACE_ID = re.compile(r"^(?:local-[a-f0-9]{16}|ws-[a-f0-9]{32})$")
_MAX_WORKSPACES = 100
_MAX_CATALOG_BYTES = 1_000_000
_SCHEMA_VERSION = 1


class WorkspaceCatalogError(ValueError):
    """The local workspace catalog is invalid or cannot be accessed safely."""


class WorkspaceNotFoundError(KeyError):
    """A requested workspace ID is not present in the local catalog."""


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    root: Path
    created_at: str
    updated_at: str
    managed: bool

    def public_dict(self, *, default_workspace_id: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_default": self.id == default_workspace_id,
            "status": "ready" if self.root.is_dir() and not self.root.is_symlink() else "missing",
        }


class WorkspaceRegistry:
    """Persist workspace names with a distinct data root for every workspace.

    The first workspace points at the existing project root, preserving legacy
    `.axiom` data. New workspaces are created below a private managed directory.
    Filesystem paths remain local and are never returned by the API.
    """

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        workspaces_root: str | Path | None = None,
        default_root: str | Path | None = None,
    ) -> None:
        self.home = Path(home).expanduser() if home is not None else Path(
            os.environ.get("AXIOM_WORKSPACES_HOME", Path.home() / ".axiom")
        ).expanduser()
        self.workspaces_root = Path(workspaces_root).expanduser() if workspaces_root is not None else Path(
            os.environ.get("AXIOM_WORKSPACES_DIR", Path.home() / "AXIOM" / "Workspaces")
        ).expanduser()
        self._provided_default_root = Path(default_root).expanduser() if default_root is not None else None
        self.catalog_file = self.home / "workspaces.json"
        self.lock_file = self.home / ".workspaces.lock"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _default_root(self) -> Path:
        candidate = self._provided_default_root or discover_workspace_root()
        if candidate.is_symlink():
            raise WorkspaceCatalogError("The default workspace root must not be a symbolic link.")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceCatalogError("The default workspace root could not be resolved.") from exc
        if not resolved.is_dir():
            raise WorkspaceCatalogError("The default workspace root must be a directory.")
        return resolved

    def _prepare_home(self) -> None:
        if self.home.is_symlink():
            raise WorkspaceCatalogError("The AXIOM workspace catalog directory must not be a symbolic link.")
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.home.is_dir():
            raise WorkspaceCatalogError("The AXIOM workspace catalog path must be a directory.")
        if self.catalog_file.is_symlink() or self.lock_file.is_symlink():
            raise WorkspaceCatalogError("The AXIOM workspace catalog files must not be symbolic links.")
        try:
            self.home.chmod(0o700)
        except OSError:
            # Windows and restricted filesystems may not expose POSIX mode changes.
            pass

    def _prepare_managed_root(self) -> Path:
        if self.workspaces_root.is_symlink():
            raise WorkspaceCatalogError("The managed workspaces directory must not be a symbolic link.")
        self.workspaces_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            resolved = self.workspaces_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceCatalogError("The managed workspaces directory could not be resolved.") from exc
        if not resolved.is_dir():
            raise WorkspaceCatalogError("The managed workspaces path must be a directory.")
        return resolved

    def _default_record(self, root: Path) -> dict[str, Any]:
        workspace_id = "local-" + hashlib.sha256(os.fsencode(root)).hexdigest()[:16]
        now = self._now()
        configured_name = os.environ.get("AXIOM_WORKSPACE_NAME", "Development").strip() or "Development"
        return {
            "id": workspace_id,
            "name": configured_name[:64],
            "root": str(root),
            "created_at": now,
            "updated_at": now,
            "managed": False,
        }

    def _validate_record(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise WorkspaceCatalogError("The AXIOM workspace catalog contains an invalid entry.")
        workspace_id = value.get("id")
        name = value.get("name")
        root_value = value.get("root")
        created_at = value.get("created_at")
        updated_at = value.get("updated_at")
        managed = value.get("managed")
        if (
            not isinstance(workspace_id, str)
            or not _WORKSPACE_ID.fullmatch(workspace_id)
            or not isinstance(name, str)
            or not name.strip()
            or len(name) > 64
            or any(ord(char) < 32 for char in name)
            or not isinstance(root_value, str)
            or len(root_value) > 4096
            or not Path(root_value).is_absolute()
            or not isinstance(created_at, str)
            or len(created_at) > 64
            or not isinstance(updated_at, str)
            or len(updated_at) > 64
            or type(managed) is not bool
        ):
            raise WorkspaceCatalogError("The AXIOM workspace catalog contains an invalid entry.")
        root = Path(root_value)
        if root.is_symlink():
            raise WorkspaceCatalogError("A workspace root must not be a symbolic link.")
        if managed:
            base = self._prepare_managed_root()
            expected = (base / workspace_id).resolve(strict=False)
            if root.resolve(strict=False) != expected:
                raise WorkspaceCatalogError("A managed workspace path escaped its designated directory.")
        try:
            resolved = root.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceCatalogError("A workspace path could not be resolved safely.") from exc
        return {
            "id": workspace_id,
            "name": name.strip(),
            "root": str(resolved),
            "created_at": created_at,
            "updated_at": updated_at,
            "managed": managed,
        }

    def _read_state(self) -> dict[str, Any]:
        if self.catalog_file.is_symlink() or self.lock_file.is_symlink():
            raise WorkspaceCatalogError("The AXIOM workspace catalog files must not be symbolic links.")
        if not self.catalog_file.exists():
            return {"schema_version": _SCHEMA_VERSION, "default_workspace_id": "", "workspaces": []}
        if not self.catalog_file.is_file() or self.catalog_file.stat().st_size > _MAX_CATALOG_BYTES:
            raise WorkspaceCatalogError("The AXIOM workspace catalog is not a bounded regular file.")
        try:
            payload = json.loads(self.catalog_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkspaceCatalogError("The AXIOM workspace catalog could not be read.") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _SCHEMA_VERSION
            or not isinstance(payload.get("default_workspace_id"), str)
            or not isinstance(payload.get("workspaces"), list)
            or len(payload["workspaces"]) > _MAX_WORKSPACES
        ):
            raise WorkspaceCatalogError("The AXIOM workspace catalog has an unsupported schema.")
        records = [self._validate_record(item) for item in payload["workspaces"]]
        ids = [item["id"] for item in records]
        if len(ids) != len(set(ids)):
            raise WorkspaceCatalogError("The AXIOM workspace catalog contains duplicate IDs.")
        default_id = payload["default_workspace_id"]
        if records and default_id not in ids:
            raise WorkspaceCatalogError("The AXIOM workspace catalog has no valid default workspace.")
        return {"schema_version": _SCHEMA_VERSION, "default_workspace_id": default_id, "workspaces": records}

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write_text(self.catalog_file, json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    def _load_state(self) -> dict[str, Any]:
        self._prepare_home()
        default_root = self._default_root()
        with file_lock(self.lock_file):
            state = self._read_state()
            records = state["workspaces"]
            default_record = next(
                (item for item in records if Path(item["root"]).resolve(strict=False) == default_root),
                None,
            )
            changed = False
            if default_record is None:
                if len(records) >= _MAX_WORKSPACES:
                    raise WorkspaceCatalogError("The workspace catalog is full; remove a workspace before adding another.")
                default_record = self._default_record(default_root)
                records.append(default_record)
                changed = True
            if state["default_workspace_id"] != default_record["id"]:
                state["default_workspace_id"] = default_record["id"]
                changed = True
            if changed or not self.catalog_file.exists():
                self._write_state(state)
            return state

    @staticmethod
    def _as_workspace(record: dict[str, Any]) -> Workspace:
        return Workspace(
            id=record["id"],
            name=record["name"],
            root=Path(record["root"]),
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            managed=record["managed"],
        )

    def list_workspaces(self) -> list[dict[str, Any]]:
        state = self._load_state()
        default_id = state["default_workspace_id"]
        records = sorted(state["workspaces"], key=lambda item: (item["id"] != default_id, item["name"].casefold()))
        return [self._as_workspace(record).public_dict(default_workspace_id=default_id) for record in records]

    def default_workspace_id(self) -> str:
        return self._load_state()["default_workspace_id"]

    def resolve(self, workspace_id: str | None = None) -> Workspace:
        state = self._load_state()
        selected_id = workspace_id or state["default_workspace_id"]
        if not isinstance(selected_id, str) or not _WORKSPACE_ID.fullmatch(selected_id):
            raise WorkspaceNotFoundError("Workspace not found.")
        record = next((item for item in state["workspaces"] if item["id"] == selected_id), None)
        if record is None:
            raise WorkspaceNotFoundError("Workspace not found.")
        workspace = self._as_workspace(record)
        if workspace.root.is_symlink():
            raise WorkspaceCatalogError("The selected workspace root must not be a symbolic link.")
        if not workspace.root.is_dir():
            raise WorkspaceCatalogError("The selected workspace directory is missing.")
        return workspace

    def create(self, name: str) -> dict[str, Any]:
        normalized = self._validate_name(name)
        self._prepare_home()
        base = self._prepare_managed_root()
        with file_lock(self.lock_file):
            state = self._read_state()
            if not state["workspaces"]:
                default = self._default_record(self._default_root())
                state["workspaces"].append(default)
                state["default_workspace_id"] = default["id"]
            if len(state["workspaces"]) >= _MAX_WORKSPACES:
                raise WorkspaceCatalogError(f"AXIOM supports at most {_MAX_WORKSPACES} workspaces.")
            if any(item["name"].casefold() == normalized.casefold() for item in state["workspaces"]):
                raise WorkspaceCatalogError("A workspace with that name already exists.")
            workspace_id = "ws-" + uuid.uuid4().hex
            root = base / workspace_id
            root.mkdir(mode=0o700)
            now = self._now()
            record = {
                "id": workspace_id,
                "name": normalized,
                "root": str(root.resolve(strict=True)),
                "created_at": now,
                "updated_at": now,
                "managed": True,
            }
            state["workspaces"].append(record)
            try:
                self._write_state(state)
            except Exception:
                root.rmdir()
                raise
        return self._as_workspace(record).public_dict(default_workspace_id=state["default_workspace_id"])

    def create_restore_staging_directory(self) -> Path:
        """Create a private same-filesystem staging directory for verified restore."""
        base = self._prepare_managed_root()
        return Path(tempfile.mkdtemp(prefix=".axiom-restore-", dir=base))

    def create_from_staging_directory(self, name: str, staging_root: str | Path) -> dict[str, Any]:
        """Atomically publish a fully populated staging tree as a new managed workspace."""
        normalized = self._validate_name(name)
        self._prepare_home()
        base = self._prepare_managed_root()
        source = Path(staging_root)
        if source.is_symlink():
            raise WorkspaceCatalogError("Restore staging directory must not be a symbolic link.")
        try:
            resolved_source = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceCatalogError("Restore staging directory is unavailable.") from exc
        if (
            not resolved_source.is_dir()
            or resolved_source.parent != base
            or not resolved_source.name.startswith(".axiom-restore-")
        ):
            raise WorkspaceCatalogError("Restore staging directory must be a direct child of the managed workspace root.")

        with file_lock(self.lock_file):
            state = self._read_state()
            if not state["workspaces"]:
                default = self._default_record(self._default_root())
                state["workspaces"].append(default)
                state["default_workspace_id"] = default["id"]
            if len(state["workspaces"]) >= _MAX_WORKSPACES:
                raise WorkspaceCatalogError(f"AXIOM supports at most {_MAX_WORKSPACES} workspaces.")
            if any(item["name"].casefold() == normalized.casefold() for item in state["workspaces"]):
                raise WorkspaceCatalogError("A workspace with that name already exists.")

            workspace_id = "ws-" + uuid.uuid4().hex
            destination = base / workspace_id
            if destination.exists() or destination.is_symlink():
                raise WorkspaceCatalogError("A managed workspace destination already exists.")
            os.rename(resolved_source, destination)
            now = self._now()
            record = {
                "id": workspace_id,
                "name": normalized,
                "root": str(destination.resolve(strict=True)),
                "created_at": now,
                "updated_at": now,
                "managed": True,
            }
            state["workspaces"].append(record)
            try:
                self._write_state(state)
            except Exception:
                try:
                    os.rename(destination, resolved_source)
                except OSError:
                    # Keep an unregistered directory rather than deleting restored user data.
                    pass
                raise
        return self._as_workspace(record).public_dict(default_workspace_id=state["default_workspace_id"])

    def rename(self, workspace_id: str, name: str) -> dict[str, Any]:
        normalized = self._validate_name(name)
        self._prepare_home()
        with file_lock(self.lock_file):
            state = self._read_state()
            record = next((item for item in state["workspaces"] if item["id"] == workspace_id), None)
            if record is None:
                raise WorkspaceNotFoundError("Workspace not found.")
            if any(item["id"] != workspace_id and item["name"].casefold() == normalized.casefold() for item in state["workspaces"]):
                raise WorkspaceCatalogError("A workspace with that name already exists.")
            record["name"] = normalized
            record["updated_at"] = self._now()
            self._write_state(state)
        return self._as_workspace(record).public_dict(default_workspace_id=state["default_workspace_id"])

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise WorkspaceCatalogError("Workspace name must be text.")
        normalized = name.strip()
        if not normalized or len(normalized) > 64 or any(ord(char) < 32 for char in normalized):
            raise WorkspaceCatalogError("Workspace name must contain 1 to 64 printable characters.")
        return normalized
