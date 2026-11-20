"""Versioned, local-first metadata for agent task profiles.

Profiles describe intended task families; they are not authorization grants and
are never used as a substitute for MCP's server-side permission policy.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from axiom.core.workspace_context import active_workspace_root
from threading import RLock
from typing import Any
from axiom.core.resource_ids import new_resource_id

SCHEMA_VERSION = 1
MAX_STORE_BYTES = 2 * 1024 * 1024
MAX_PROFILES = 100
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")
TASK_CAPABILITIES = (
    "model.inspect",
    "dataset.profile",
    "training.plan",
    "evaluation.review",
    "runtime.inspect",
    "mcp.inspect",
)
_LOCK = RLock()


class AgentProfileError(ValueError):
    """Invalid or unsafe local agent-profile store operation."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_profile() -> dict[str, Any]:
    now = _now()
    return {
        "id": "default",
        "name": "default",
        "version": 1,
        "task_capabilities": [],
        "created_at": now,
        "updated_at": now,
        "built_in": True,
    }


def validate_capabilities(capabilities: list[str]) -> list[str]:
    if len(capabilities) > len(TASK_CAPABILITIES):
        raise AgentProfileError("Too many task capabilities were supplied.")
    if any(capability not in TASK_CAPABILITIES for capability in capabilities):
        raise AgentProfileError("One or more task capabilities are not in the supported catalog.")
    if len(set(capabilities)) != len(capabilities):
        raise AgentProfileError("Task capabilities must be unique.")
    return sorted(capabilities)


class AgentProfileStore:
    """Store versioned profile metadata under the current workspace's `.axiom`."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or active_workspace_root()).resolve()
        self.directory = self.root / ".axiom"
        self.path = self.directory / "agent-profiles.json"

    def _check_paths(self, *, create: bool = False) -> None:
        if self.directory.is_symlink():
            raise AgentProfileError("The .axiom directory must not be a symbolic link.")
        if self.directory.exists() and not self.directory.is_dir():
            raise AgentProfileError("The .axiom path is not a directory.")
        if create:
            self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        if self.directory.exists() and not self.directory.resolve().is_relative_to(self.root):
            raise AgentProfileError("The agent-profile directory escapes the workspace.")
        if self.path.is_symlink():
            raise AgentProfileError("The agent-profile file must not be a symbolic link.")

    def load(self) -> dict[str, Any]:
        with _LOCK:
            self._check_paths()
            if not self.path.exists():
                return {"schema_version": SCHEMA_VERSION, "profiles": [_default_profile()]}
            try:
                if self.path.stat().st_size > MAX_STORE_BYTES:
                    raise AgentProfileError("The agent-profile store exceeds the permitted size.")
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except AgentProfileError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise AgentProfileError("The agent-profile store could not be read safely.") from exc
            return self._validate_document(raw)

    def _validate_document(self, raw: Any) -> dict[str, Any]:
        schema_version = raw.get("schema_version") if isinstance(raw, dict) else None
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != SCHEMA_VERSION:
            raise AgentProfileError("Unsupported agent-profile schema version.")
        profiles = raw.get("profiles")
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= MAX_PROFILES:
            raise AgentProfileError("The agent-profile store has an invalid profile list.")
        normalized: list[dict[str, Any]] = []
        ids: set[str] = set()
        names: set[str] = set()
        for item in profiles:
            if not isinstance(item, dict):
                raise AgentProfileError("The agent-profile store contains an invalid entry.")
            profile_id = item.get("id")
            name = item.get("name")
            version = item.get("version")
            caps = item.get("task_capabilities")
            if not isinstance(profile_id, str) or not (profile_id == "default" or re.fullmatch(r"[a-f0-9]{32}", profile_id)):
                raise AgentProfileError("The agent-profile store contains an invalid ID.")
            if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name):
                raise AgentProfileError("The agent-profile store contains an invalid name.")
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise AgentProfileError("The agent-profile store contains an invalid profile version.")
            if not isinstance(caps, list) or not all(isinstance(cap, str) for cap in caps):
                raise AgentProfileError("The agent-profile store contains invalid task capabilities.")
            if profile_id in ids or name.casefold() in names:
                raise AgentProfileError("Agent-profile IDs and names must be unique.")
            validate_capabilities(caps)
            if profile_id == "default" and (name != "default" or caps):
                raise AgentProfileError("The default profile must remain named default with no task capabilities.")
            ids.add(profile_id)
            names.add(name.casefold())
            normalized.append({
                "id": profile_id,
                "name": name,
                "version": version,
                "task_capabilities": sorted(caps),
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", "")),
                "built_in": profile_id == "default",
            })
        if "default" not in ids:
            raise AgentProfileError("The required default-deny profile is missing.")
        return {"schema_version": SCHEMA_VERSION, "profiles": normalized}

    def _save(self, document: dict[str, Any]) -> None:
        self._check_paths(create=True)
        payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if len(payload) > MAX_STORE_BYTES:
            raise AgentProfileError("The agent-profile store exceeds the permitted size.")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".agent-profiles-", suffix=".tmp", dir=self.directory, delete=False) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError as exc:
                raise AgentProfileError(
                    "Could not secure agent-profile file permissions."
                ) from exc
            if self.path.is_symlink():
                raise AgentProfileError("The agent-profile file must not be a symbolic link.")
            os.replace(temporary, self.path)
            temporary = None
        except AgentProfileError:
            raise
        except OSError as exc:
            raise AgentProfileError("The agent-profile store could not be saved safely.") from exc
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    # Cleanup must not hide the original save failure.
                    pass

    def create(self, name: str, capabilities: list[str] | None = None) -> dict[str, Any]:
        normalized_name = name.strip()
        if not PROFILE_NAME.fullmatch(normalized_name):
            raise AgentProfileError("Profile names must be 1–64 safe characters and start with a letter or number.")
        caps = validate_capabilities(capabilities or [])
        with _LOCK:
            document = self.load()
            if len(document["profiles"]) >= MAX_PROFILES:
                raise AgentProfileError("The profile limit has been reached.")
            if any(profile["name"].casefold() == normalized_name.casefold() for profile in document["profiles"]):
                raise AgentProfileError("A profile with that name already exists.")
            now = _now()
            profile = {
                "id": new_resource_id(),
                "name": normalized_name,
                "version": 1,
                "task_capabilities": caps,
                "created_at": now,
                "updated_at": now,
                "built_in": False,
            }
            document["profiles"].append(profile)
            self._save(document)
            return profile

    def update_capabilities(self, profile_id: str, capabilities: list[str]) -> dict[str, Any]:
        caps = validate_capabilities(capabilities)
        with _LOCK:
            document = self.load()
            profile = next((item for item in document["profiles"] if item["id"] == profile_id), None)
            if profile is None:
                raise AgentProfileError("Agent profile not found.")
            profile["task_capabilities"] = caps
            profile["version"] += 1
            profile["updated_at"] = _now()
            self._save(document)
            return profile

    def delete(self, profile_id: str) -> dict[str, Any]:
        with _LOCK:
            document = self.load()
            if profile_id == "default":
                raise AgentProfileError("The default profile cannot be deleted.")
            remaining = [item for item in document["profiles"] if item["id"] != profile_id]
            if len(remaining) == len(document["profiles"]):
                raise AgentProfileError("Agent profile not found.")
            document["profiles"] = remaining
            self._save(document)
            return {"deleted_id": profile_id, "remaining": len(remaining)}
