"""Strict declarative plugin manifests and hash-bound default-deny grants.

Manifests are metadata only. They cannot define commands, entry points, code,
URLs, or executable hooks, and this store never loads plugin code.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir
from axiom.path_safety import open_regular_file

PLUGIN_MANIFEST_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MANIFESTS = 50
MAX_TOOLS_PER_MANIFEST = 32
MAX_PLUGIN_STORE_BYTES = 4 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 1024
MAX_TOOL_DESCRIPTION_CHARS = 2048

_PLUGIN_ID = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+){1,7}$")
_PLUGIN_VERSION = re.compile(
    r"^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?"
    r"(?:\\+[0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*)?$"
)
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ALLOWED_SCOPES = frozenset({"read", "write", "execution", "network", "secrets"})


class PluginManifestError(ValueError):
    """A plugin manifest or permission record failed strict validation."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Plugin registry JSON contains duplicate object keys.")
        result[key] = value
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise PluginManifestError("Plugin manifest contains unsupported JSON values.") from None


def _text(value: Any, *, label: str, maximum: int, minimum: int = 1) -> str:
    if not isinstance(value, str):
        raise PluginManifestError(f"{label} must be a string.")
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        raise PluginManifestError(f"{label} must contain {minimum} to {maximum} characters.")
    if any(not character.isprintable() for character in value):
        raise PluginManifestError(f"{label} must not contain control characters.")
    return value


def _version_key(version: str) -> tuple[Any, ...]:
    match = _PLUGIN_VERSION.fullmatch(version)
    if match is None:
        raise PluginManifestError("Plugin version must use semantic versioning (MAJOR.MINOR.PATCH).")
    major, minor, patch, prerelease = match.groups()
    if prerelease is None:
        pre_key: tuple[Any, ...] = (1,)
    else:
        identifiers = prerelease.split(".")
        if any(not item for item in identifiers):
            raise PluginManifestError("Plugin semantic-version prerelease is invalid.")
        if any(item.isdigit() and len(item) > 1 and item.startswith("0") for item in identifiers):
            raise PluginManifestError("Numeric prerelease identifiers must not contain leading zeroes.")
        pre_key = (0, *(
            (0, int(item)) if item.isdigit() else (1, item)
            for item in identifiers
        ))
    return int(major), int(minor), int(patch), pre_key


def _version_is_newer(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def validate_plugin_manifest(document: Any) -> tuple[dict[str, Any], str]:
    """Validate one declarative schema-v1 manifest and compute its canonical SHA-256."""
    if not isinstance(document, dict):
        raise PluginManifestError("Plugin manifest must be a JSON object.")
    if len(_canonical_json(document)) > MAX_MANIFEST_BYTES:
        raise PluginManifestError(f"Plugin manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit.")
    expected_keys = {"manifest_schema_version", "id", "name", "version", "description", "tools"}
    if set(document) != expected_keys:
        raise PluginManifestError("Plugin manifest contains missing or unsupported fields.")
    schema_version = document.get("manifest_schema_version")
    if type(schema_version) is not int or schema_version != PLUGIN_MANIFEST_SCHEMA_VERSION:
        raise PluginManifestError("Unsupported plugin manifest schema version.")
    plugin_id = _text(document.get("id"), label="Plugin ID", maximum=160)
    if not _PLUGIN_ID.fullmatch(plugin_id):
        raise PluginManifestError("Plugin ID must be a lowercase reverse-DNS style identifier.")
    name = _text(document.get("name"), label="Plugin name", maximum=96)
    version = _text(document.get("version"), label="Plugin version", maximum=64)
    _version_key(version)
    description = _text(
        document.get("description"), label="Plugin description", maximum=MAX_DESCRIPTION_CHARS
    )
    tools = document.get("tools")
    if not isinstance(tools, list) or not 1 <= len(tools) <= MAX_TOOLS_PER_MANIFEST:
        raise PluginManifestError(f"Plugin manifest must declare 1 to {MAX_TOOLS_PER_MANIFEST} tools.")
    normalized_tools: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in tools:
        if not isinstance(item, dict) or set(item) != {"name", "description", "scopes"}:
            raise PluginManifestError("Every plugin tool must contain only name, description, and scopes.")
        tool_name = _text(item.get("name"), label="Tool name", maximum=64)
        if not _TOOL_NAME.fullmatch(tool_name) or tool_name in seen_names:
            raise PluginManifestError("Plugin tool names must be unique, lowercase identifiers.")
        seen_names.add(tool_name)
        tool_description = _text(
            item.get("description"),
            label=f"Description for tool {tool_name}",
            maximum=MAX_TOOL_DESCRIPTION_CHARS,
        )
        scopes = item.get("scopes")
        if (
            not isinstance(scopes, list)
            or not 1 <= len(scopes) <= len(_ALLOWED_SCOPES)
            or any(not isinstance(scope, str) or scope not in _ALLOWED_SCOPES for scope in scopes)
            or len(set(scopes)) != len(scopes)
        ):
            raise PluginManifestError(f"Tool {tool_name} has invalid or duplicate permission scopes.")
        normalized_tools.append({
            "name": tool_name,
            "description": tool_description,
            "scopes": sorted(scopes),
        })
    normalized = {
        "manifest_schema_version": PLUGIN_MANIFEST_SCHEMA_VERSION,
        "id": plugin_id,
        "name": name,
        "version": version,
        "description": description,
        "tools": sorted(normalized_tools, key=lambda item: item["name"]),
    }
    encoded = _canonical_json(normalized)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise PluginManifestError(f"Plugin manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit.")
    return normalized, hashlib.sha256(encoded).hexdigest()


class PluginManifestStore:
    """Workspace-local manifest inventory with explicit hash-bound grants."""

    def __init__(self, root: Path | None = None) -> None:
        metadata_root = Path(root) if root is not None else workspace_metadata_dir()
        self.root = metadata_root / "plugins"
        self.registry_file = self.root / "registry.json"
        self.lock_file = self.root / ".registry.lock"

    def _assert_safe_paths(self, *, require_registry: bool = False) -> None:
        for path in (self.root.parent, self.root, self.registry_file, self.lock_file):
            if path.is_symlink():
                raise PluginManifestError("Plugin manifest store paths must not contain symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise PluginManifestError("Plugin manifest store must be a directory.")
        if self.registry_file.exists() and not self.registry_file.is_file():
            raise PluginManifestError("Plugin manifest registry must be a regular file.")
        if self.registry_file.exists():
            try:
                if self.registry_file.stat().st_size > MAX_PLUGIN_STORE_BYTES:
                    raise PluginManifestError("Plugin manifest registry exceeds its storage limit.")
            except OSError:
                raise PluginManifestError("Plugin manifest registry could not be inspected safely.") from None
        elif require_registry:
            raise PluginManifestError("Plugin manifest registry does not exist.")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"store_schema_version": 1, "manifests": [], "grants": {}}

    def _read(self) -> dict[str, Any]:
        self._assert_safe_paths()
        try:
            with open_regular_file(self.registry_file) as stream:
                info = __import__("os").fstat(stream.fileno())
                if not __import__("stat").S_ISREG(info.st_mode) or info.st_size > MAX_PLUGIN_STORE_BYTES:
                    raise PluginManifestError("Plugin manifest registry is not a bounded regular file.")
                raw_bytes = stream.read(MAX_PLUGIN_STORE_BYTES + 1)
            if len(raw_bytes) > MAX_PLUGIN_STORE_BYTES:
                raise PluginManifestError("Plugin manifest registry exceeds its storage limit.")
            document = json.loads(
                raw_bytes.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except FileNotFoundError:
            return self._empty()
        except PluginManifestError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raise PluginManifestError("Plugin manifest registry could not be read safely.") from None

        if (
            not isinstance(document, dict)
            or set(document) != {"store_schema_version", "manifests", "grants"}
            or type(document.get("store_schema_version")) is not int
            or document["store_schema_version"] != 1
            or not isinstance(document.get("manifests"), list)
            or len(document["manifests"]) > MAX_MANIFESTS
            or not isinstance(document.get("grants"), dict)
        ):
            raise PluginManifestError("Plugin manifest registry has an invalid schema.")
        manifest_ids: set[str] = set()
        normalized_manifests: list[dict[str, Any]] = []
        for record in document["manifests"]:
            if not isinstance(record, dict) or set(record) != {"manifest", "sha256"}:
                raise PluginManifestError("Plugin manifest registry contains an invalid record.")
            manifest, digest = validate_plugin_manifest(record["manifest"])
            if record["sha256"] != digest or manifest["id"] in manifest_ids:
                raise PluginManifestError("Plugin manifest digest or identity is invalid.")
            manifest_ids.add(manifest["id"])
            normalized_manifests.append({"manifest": manifest, "sha256": digest})

        for plugin_id, grant in document["grants"].items():
            if (
                not isinstance(plugin_id, str)
                or plugin_id not in manifest_ids
                or not isinstance(grant, dict)
                or set(grant) != {"manifest_sha256", "tools"}
                or not isinstance(grant.get("manifest_sha256"), str)
                or not _SHA256.fullmatch(grant["manifest_sha256"])
                or not isinstance(grant.get("tools"), list)
                or len(grant["tools"]) > MAX_TOOLS_PER_MANIFEST
            ):
                raise PluginManifestError("Plugin grant registry contains an invalid record.")
            current_record = next(
                record for record in normalized_manifests
                if record["manifest"]["id"] == plugin_id
            )
            stale_grant = grant["manifest_sha256"] != current_record["sha256"]
            declared = {
                tool["name"]: set(tool["scopes"])
                for tool in current_record["manifest"]["tools"]
            }
            seen_names: set[str] = set()
            for tool_grant in grant["tools"]:
                if (
                    not isinstance(tool_grant, dict)
                    or set(tool_grant) != {"name", "scopes"}
                    or not isinstance(tool_grant.get("name"), str)
                    or not _TOOL_NAME.fullmatch(tool_grant["name"])
                    or tool_grant["name"] in seen_names
                    or not isinstance(tool_grant.get("scopes"), list)
                    or not tool_grant["scopes"]
                    or any(
                        not isinstance(scope, str) or scope not in _ALLOWED_SCOPES
                        for scope in tool_grant["scopes"]
                    )
                    or len(set(tool_grant["scopes"])) != len(tool_grant["scopes"])
                    or (
                        not stale_grant
                        and (
                            tool_grant["name"] not in declared
                            or any(
                                scope not in declared[tool_grant["name"]]
                                for scope in tool_grant["scopes"]
                            )
                        )
                    )
                ):
                    raise PluginManifestError("Plugin grant contains an unknown tool or undeclared scope.")
                seen_names.add(tool_grant["name"])

        return {
            "store_schema_version": 1,
            "manifests": normalized_manifests,
            "grants": document["grants"],
        }

    def _write(self, document: dict[str, Any]) -> None:
        self._assert_safe_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if len(serialized.encode("utf-8")) > MAX_PLUGIN_STORE_BYTES:
            raise PluginManifestError("Plugin manifest registry exceeds its storage limit.")
        atomic_write_text(self.registry_file, serialized)

    def list(self) -> dict[str, Any]:
        document = self._read()
        manifests = []
        for record in document["manifests"]:
            manifest = record["manifest"]
            grant = document["grants"].get(manifest["id"])
            if not grant or not grant["tools"]:
                status = "default_deny"
            elif grant["manifest_sha256"] != record["sha256"]:
                status = "stale_manifest"
            else:
                status = "granted"
            manifests.append({
                **record,
                "permission_status": status,
                "granted_tools": grant["tools"] if grant and status == "granted" else [],
            })
        return {
            "store_schema_version": 1,
            "manifests": manifests,
            "maximum_manifests": MAX_MANIFESTS,
            "maximum_tools_per_manifest": MAX_TOOLS_PER_MANIFEST,
            "execution_note": (
                "Manifests are declarative metadata and permission records only. "
                "AXIOM does not load plugin code or execute manifest-defined commands."
            ),
        }

    def register(self, document: Any) -> dict[str, Any]:
        manifest, digest = validate_plugin_manifest(document)
        self._assert_safe_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            stored = self._read()
            existing_index = next(
                (
                    index for index, record in enumerate(stored["manifests"])
                    if record["manifest"]["id"] == manifest["id"]
                ),
                None,
            )
            if existing_index is None:
                if len(stored["manifests"]) >= MAX_MANIFESTS:
                    raise PluginManifestError(f"A workspace can register at most {MAX_MANIFESTS} plugins.")
                stored["manifests"].append({"manifest": manifest, "sha256": digest})
            else:
                previous = stored["manifests"][existing_index]["manifest"]
                if not _version_is_newer(manifest["version"], previous["version"]):
                    raise PluginManifestError(
                        "A registered plugin can only be updated to a strictly newer semantic version."
                    )
                stored["manifests"][existing_index] = {"manifest": manifest, "sha256": digest}
            stored["manifests"].sort(key=lambda item: item["manifest"]["id"])
            self._write(stored)
        return {"manifest": manifest, "sha256": digest, "permission_status": "default_deny"}

    def grant(
        self,
        plugin_id: str,
        *,
        expected_sha256: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(plugin_id, str) or not _PLUGIN_ID.fullmatch(plugin_id):
            raise PluginManifestError("Plugin ID is invalid.")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
            raise PluginManifestError("The current manifest SHA-256 is required to grant permissions.")
        if not isinstance(tools, list) or len(tools) > MAX_TOOLS_PER_MANIFEST:
            raise PluginManifestError("Plugin grant tool list is invalid.")
        self._assert_safe_paths(require_registry=True)
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            stored = self._read()
            record = next(
                (item for item in stored["manifests"] if item["manifest"]["id"] == plugin_id),
                None,
            )
            if record is None:
                raise PluginManifestError("Plugin manifest is not registered.")
            if record["sha256"] != expected_sha256:
                raise PluginManifestError("Plugin manifest changed; review the new version before granting permissions.")
            declared = {
                tool["name"]: set(tool["scopes"])
                for tool in record["manifest"]["tools"]
            }
            normalized_grants = []
            seen: set[str] = set()
            for item in tools:
                if not isinstance(item, dict) or set(item) != {"name", "scopes"}:
                    raise PluginManifestError("Plugin grant entries must contain only name and scopes.")
                name = _text(item["name"], label="Granted tool name", maximum=64)
                scopes = item["scopes"]
                if (
                    name not in declared
                    or name in seen
                    or not isinstance(scopes, list)
                    or not scopes
                    or any(not isinstance(scope, str) or scope not in declared[name] for scope in scopes)
                    or len(set(scopes)) != len(scopes)
                ):
                    raise PluginManifestError(
                        "Grant must use unique declared tools and a subset of their manifest scopes."
                    )
                seen.add(name)
                normalized_grants.append({"name": name, "scopes": sorted(scopes)})
            stored["grants"][plugin_id] = {
                "manifest_sha256": expected_sha256,
                "tools": sorted(normalized_grants, key=lambda item: item["name"]),
            }
            self._write(stored)
        return {
            "plugin_id": plugin_id,
            "manifest_sha256": expected_sha256,
            "tools": normalized_grants,
        }

    def revoke(self, plugin_id: str) -> bool:
        if not isinstance(plugin_id, str) or not _PLUGIN_ID.fullmatch(plugin_id):
            raise PluginManifestError("Plugin ID is invalid.")
        self._assert_safe_paths()
        if not self.root.exists():
            return False
        with file_lock(self.lock_file):
            stored = self._read()
            existed = stored["grants"].pop(plugin_id, None) is not None
            if existed:
                self._write(stored)
            return existed

    def is_allowed(self, plugin_id: str, tool_name: str, required_scopes: list[str]) -> bool:
        if (
            not isinstance(plugin_id, str)
            or not _PLUGIN_ID.fullmatch(plugin_id)
            or not isinstance(tool_name, str)
            or not _TOOL_NAME.fullmatch(tool_name)
            or not isinstance(required_scopes, list)
            or not required_scopes
            or any(not isinstance(scope, str) or scope not in _ALLOWED_SCOPES for scope in required_scopes)
            or len(set(required_scopes)) != len(required_scopes)
        ):
            return False
        document = self._read()
        record = next(
            (item for item in document["manifests"] if item["manifest"]["id"] == plugin_id),
            None,
        )
        grant = document["grants"].get(plugin_id)
        if record is None or grant is None or grant["manifest_sha256"] != record["sha256"]:
            return False
        allowed = next((item for item in grant["tools"] if item["name"] == tool_name), None)
        if allowed is None:
            return False
        return set(required_scopes).issubset(allowed["scopes"])
