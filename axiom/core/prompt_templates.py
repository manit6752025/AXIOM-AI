"""Workspace-local, versioned prompt templates with strictly typed substitution.

Rendering is a deterministic text substitution only. It never evaluates
expressions, runs tools, invokes a model, or interprets template code.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import active_workspace_root

SCHEMA_VERSION = 1
MAX_STORE_BYTES = 8 * 1024 * 1024
MAX_TEMPLATES = 100
MAX_REVISIONS = 20
MAX_BODY_CHARS = 64 * 1024
MAX_RENDERED_CHARS = 128 * 1024
MAX_VARIABLES = 32
MAX_VALUE_CHARS = 16 * 1024
MAX_CHOICES = 32
MAX_NAME_CHARS = 80
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_VARIABLE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_TEMPLATE_ID = re.compile(r"^[0-9a-f]{32}$")
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")
_LOCK_NAME = ".prompt-templates.lock"
_VARIABLE_TYPES = frozenset({"string", "integer", "number", "boolean", "choice"})
_VARIABLE_FIELDS = {
    "name", "type", "required", "default", "description", "choices",
    "min_length", "max_length", "minimum", "maximum",
}
_TEMPLATE_CONTENT_FIELDS = {"name", "description", "body", "variables"}


class PromptTemplateError(ValueError):
    """Invalid, missing, stale or unsafe prompt template operation."""


class PromptTemplateConflict(PromptTemplateError):
    """Optimistic concurrency conflict while updating a template."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 40:
        raise PromptTemplateError("A prompt template timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
    except ValueError:
        raise PromptTemplateError("A prompt template timestamp must include a timezone.") from None
    return parsed.astimezone(UTC).isoformat()


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PromptTemplateError(f"Prompt template {label} must be text.")
    if len(value) > maximum or (not allow_empty and not value.strip()):
        raise PromptTemplateError(f"Prompt template {label} is empty or exceeds its {maximum}-character limit.")
    if any((ord(char) < 32 and char not in "\n\r\t") or ord(char) == 127 for char in value):
        raise PromptTemplateError(f"Prompt template {label} contains unsupported control characters.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise PromptTemplateError(f"Prompt template {label} contains invalid Unicode.") from None
    if len(encoded) > maximum * 4:
        raise PromptTemplateError(f"Prompt template {label} exceeds its UTF-8 size limit.")
    return value


def _typed_value(spec: dict[str, Any], value: Any, *, label: str) -> Any:
    kind = spec["type"]
    if kind == "string":
        if not isinstance(value, str):
            raise PromptTemplateError(f"Prompt variable {label} must be a string.")
        if len(value) > MAX_VALUE_CHARS:
            raise PromptTemplateError(f"Prompt variable {label} exceeds its value-size limit.")
        if spec["min_length"] is not None and len(value) < spec["min_length"]:
            raise PromptTemplateError(f"Prompt variable {label} is shorter than its minimum length.")
        if spec["max_length"] is not None and len(value) > spec["max_length"]:
            raise PromptTemplateError(f"Prompt variable {label} exceeds its maximum length.")
        return value
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise PromptTemplateError(f"Prompt variable {label} must be an integer.")
        number = value
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PromptTemplateError(f"Prompt variable {label} must be a finite number.")
        number = float(value)
    elif kind == "boolean":
        if type(value) is not bool:
            raise PromptTemplateError(f"Prompt variable {label} must be a boolean.")
        return value
    elif kind == "choice":
        if not isinstance(value, str) or value not in spec["choices"]:
            raise PromptTemplateError(f"Prompt variable {label} must match one of its allowed choices.")
        return value
    else:
        raise PromptTemplateError("A prompt template variable has an unsupported type.")
    if spec["minimum"] is not None and number < spec["minimum"]:
        raise PromptTemplateError(f"Prompt variable {label} is below its minimum.")
    if spec["maximum"] is not None and number > spec["maximum"]:
        raise PromptTemplateError(f"Prompt variable {label} exceeds its maximum.")
    return number


def _normalize_variables(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > MAX_VARIABLES:
        raise PromptTemplateError(f"A prompt template must have no more than {MAX_VARIABLES} variables.")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) - _VARIABLE_FIELDS:
            raise PromptTemplateError("A prompt variable contains unsupported fields.")
        name = item.get("name")
        kind = item.get("type")
        required = item.get("required", True)
        if not isinstance(name, str) or not _VARIABLE_NAME.fullmatch(name):
            raise PromptTemplateError("Prompt variable names must use a letter followed by letters, digits or underscores.")
        if name in names:
            raise PromptTemplateError("Prompt variable names must be unique.")
        if not isinstance(kind, str) or kind not in _VARIABLE_TYPES:
            raise PromptTemplateError("Prompt variable type must be string, integer, number, boolean or choice.")
        if type(required) is not bool:
            raise PromptTemplateError(f"Prompt variable {name} required must be a boolean.")
        description = item.get("description", "")
        if not isinstance(description, str) or len(description) > 256:
            raise PromptTemplateError(f"Prompt variable {name} description is invalid.")
        choices = item.get("choices", [])
        if kind == "choice":
            if not isinstance(choices, list) or not 1 <= len(choices) <= MAX_CHOICES:
                raise PromptTemplateError(f"Choice variable {name} must define 1–{MAX_CHOICES} choices.")
            if any(not isinstance(choice, str) or not choice or len(choice) > 128 for choice in choices):
                raise PromptTemplateError(f"Choice variable {name} contains an invalid choice.")
            if len(set(choices)) != len(choices):
                raise PromptTemplateError(f"Choice variable {name} has duplicate choices.")
        elif choices not in (None, []):
            raise PromptTemplateError(f"Only choice variables may define choices ({name}).")
        min_length, max_length = item.get("min_length"), item.get("max_length")
        minimum, maximum = item.get("minimum"), item.get("maximum")
        if kind == "string":
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_VALUE_CHARS)
                   for value in (min_length, max_length)):
                raise PromptTemplateError(f"String variable {name} has invalid length constraints.")
            if min_length is not None and max_length is not None and min_length > max_length:
                raise PromptTemplateError(f"String variable {name} has inconsistent length constraints.")
            if minimum is not None or maximum is not None:
                raise PromptTemplateError(f"String variable {name} cannot define numeric bounds.")
        elif kind in {"integer", "number"}:
            if min_length is not None or max_length is not None:
                raise PromptTemplateError(f"Numeric variable {name} cannot define string length constraints.")
            for value in (minimum, maximum):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                    raise PromptTemplateError(f"Numeric variable {name} has an invalid bound.")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise PromptTemplateError(f"Numeric variable {name} has inconsistent bounds.")
            if kind == "integer" and any(value is not None and not isinstance(value, int) for value in (minimum, maximum)):
                raise PromptTemplateError(f"Integer variable {name} requires integer bounds.")
        else:
            if any(value is not None for value in (min_length, max_length, minimum, maximum)):
                raise PromptTemplateError(f"Variable {name} cannot define length or numeric bounds.")
        spec = {
            "name": name,
            "type": kind,
            "required": required,
            "default": item.get("default"),
            "description": description,
            "choices": list(choices or []),
            "min_length": min_length,
            "max_length": max_length,
            "minimum": minimum,
            "maximum": maximum,
        }
        if spec["default"] is not None:
            spec["default"] = _typed_value(spec, spec["default"], label=name)
        names.add(name)
        normalized.append(spec)
    return normalized


def _normalize_content(name: Any, description: Any, body: Any, variables: Any) -> dict[str, Any]:
    safe_name = _text(name, "name", MAX_NAME_CHARS)
    if not _NAME.fullmatch(safe_name.strip()):
        raise PromptTemplateError("Prompt template names must start with a letter or number and contain only safe characters.")
    safe_description = _text(description if description is not None else "", "description", 240, allow_empty=True)
    safe_body = _text(body, "body", MAX_BODY_CHARS)
    safe_variables = _normalize_variables(variables)
    by_name = {item["name"] for item in safe_variables}
    matches = _PLACEHOLDER.findall(safe_body)
    residue = _PLACEHOLDER.sub("", safe_body)
    if "{{" in residue or "}}" in residue:
        raise PromptTemplateError("Prompt placeholders must use only {{variable_name}} syntax.")
    unknown = sorted(set(matches) - by_name)
    if unknown:
        raise PromptTemplateError("Prompt template references an undefined variable: " + ", ".join(unknown))
    return {
        "name": safe_name.strip(),
        "description": safe_description,
        "body": safe_body,
        "variables": safe_variables,
    }


def _content_digest(content: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(content)).hexdigest()


def _snapshot(record: dict[str, Any]) -> dict[str, Any]:
    content = {key: record[key] for key in _TEMPLATE_CONTENT_FIELDS}
    return {
        "version": record["version"],
        "updated_at": record["updated_at"],
        **content,
        "content_sha256": _content_digest(content),
    }


def _validate_snapshot(raw: Any) -> dict[str, Any]:
    fields = {"version", "updated_at", *_TEMPLATE_CONTENT_FIELDS, "content_sha256"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise PromptTemplateError("A retained prompt revision is malformed.")
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 1_000_000:
        raise PromptTemplateError("A retained prompt revision has an invalid version.")
    updated_at = _timestamp(raw.get("updated_at"))
    content = _normalize_content(raw.get("name"), raw.get("description"), raw.get("body"), raw.get("variables"))
    digest = raw.get("content_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != _content_digest(content):
        raise PromptTemplateError("A retained prompt revision fingerprint does not match its content.")
    return {"version": version, "updated_at": updated_at, **content, "content_sha256": digest}


def _validate_template(raw: Any) -> dict[str, Any]:
    fields = {"id", "version", "created_at", "updated_at", "revisions", *_TEMPLATE_CONTENT_FIELDS, "content_sha256"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise PromptTemplateError("A prompt template record has unsupported fields.")
    identity = raw.get("id")
    if not isinstance(identity, str) or not _TEMPLATE_ID.fullmatch(identity):
        raise PromptTemplateError("A prompt template ID is invalid.")
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 1_000_000:
        raise PromptTemplateError("A prompt template version is invalid.")
    created_at, updated_at = _timestamp(raw.get("created_at")), _timestamp(raw.get("updated_at"))
    content = _normalize_content(raw.get("name"), raw.get("description"), raw.get("body"), raw.get("variables"))
    digest = raw.get("content_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != _content_digest(content):
        raise PromptTemplateError("A prompt template content fingerprint does not match its content.")
    revisions = raw.get("revisions")
    if not isinstance(revisions, list) or len(revisions) > MAX_REVISIONS:
        raise PromptTemplateError("Prompt template revision history exceeds its limit.")
    safe_revisions = [_validate_snapshot(item) for item in revisions]
    versions = [item["version"] for item in safe_revisions]
    if versions != sorted(set(versions)) or (versions and versions[-1] >= version):
        raise PromptTemplateError("Prompt template revision history has invalid ordering.")
    if versions and (
        versions[-1] != version - 1
        or versions[0] != version - len(versions)
        or any(right != left + 1 for left, right in zip(versions, versions[1:]))
    ):
        raise PromptTemplateError("Prompt template revision history is missing or misorders a predecessor.")
    return {
        "id": identity,
        "version": version,
        "created_at": created_at,
        "updated_at": updated_at,
        **content,
        "content_sha256": digest,
        "revisions": safe_revisions,
    }


def _render_values(variables: list[dict[str, Any]], values: Any) -> dict[str, str]:
    if not isinstance(values, dict) or len(values) > MAX_VARIABLES or any(not isinstance(key, str) or not _VARIABLE_NAME.fullmatch(key) for key in values):
        raise PromptTemplateError("Prompt render values must be an object containing no more than 32 named variables.")
    if len(_canonical(values)) > MAX_VARIABLES * MAX_VALUE_CHARS:
        raise PromptTemplateError("Prompt render values exceed the total input-size limit.")
    declared = {item["name"]: item for item in variables}
    extras = sorted(set(values) - set(declared))
    if extras:
        raise PromptTemplateError("Prompt render supplied unknown variables: " + ", ".join(extras))
    resolved: dict[str, str] = {}
    for name, spec in declared.items():
        if name in values:
            value = _typed_value(spec, values[name], label=name)
        elif spec["default"] is not None:
            value = spec["default"]
        elif spec["required"]:
            raise PromptTemplateError(f"Prompt render is missing required variable: {name}")
        else:
            value = ""
        if spec["type"] == "boolean":
            rendered = "true" if value else "false"
        elif spec["type"] == "number":
            rendered = format(value, ".15g")
        else:
            rendered = str(value)
        resolved[name] = rendered
    return resolved


def render_prompt(body: str, variables: list[dict[str, Any]], values: dict[str, Any]) -> str:
    """Render exact {{name}} placeholders using validated JSON values only."""
    resolved = _render_values(variables, values)
    names = {item["name"] for item in variables}
    matches = _PLACEHOLDER.findall(body)
    if set(matches) - names:
        raise PromptTemplateError("Prompt template references an undefined variable.")
    rendered = _PLACEHOLDER.sub(lambda match: resolved[match.group(1)], body)
    if "{{" in rendered or "}}" in rendered:
        raise PromptTemplateError("Rendered prompt contains an unresolved placeholder.")
    if len(rendered) > MAX_RENDERED_CHARS or len(rendered.encode("utf-8")) > MAX_RENDERED_CHARS * 4:
        raise PromptTemplateError("Rendered prompt exceeds its output-size limit.")
    return rendered


class PromptTemplateStore:
    """Store local prompt templates and bounded revisions in the active workspace."""
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else active_workspace_root()
        self.directory = self.root / ".axiom"
        self.path = self.directory / "prompt-templates.json"
        self.lock_path = self.directory / _LOCK_NAME

    def _check_paths(self, *, create: bool = False) -> None:
        if self.root.is_symlink() or self.directory.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise PromptTemplateError("Prompt template storage must not use symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise PromptTemplateError("The active workspace is not a directory.")
        if self.directory.exists() and not self.directory.is_dir():
            raise PromptTemplateError("The workspace .axiom path is not a directory.")
        if create:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.exists() and not self.directory.resolve().is_relative_to(self.root.resolve()):
            raise PromptTemplateError("Prompt template storage escapes the active workspace.")
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise PromptTemplateError("Prompt template storage paths must not be symbolic links.")

    def _read(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.path.exists():
            return []
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise PromptTemplateError("The prompt template store exceeds its size or file-type limit.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except PromptTemplateError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise PromptTemplateError("The prompt template store could not be read safely.") from None
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "templates"} or type(raw.get("schema_version")) is not int or raw["schema_version"] != SCHEMA_VERSION:
            raise PromptTemplateError("The prompt template store has an unsupported schema.")
        records = raw.get("templates")
        if not isinstance(records, list) or len(records) > MAX_TEMPLATES:
            raise PromptTemplateError("The prompt template store exceeds its record limit.")
        normalized = [_validate_template(item) for item in records]
        ids = [item["id"] for item in normalized]
        names = [item["name"].casefold() for item in normalized]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise PromptTemplateError("Prompt template IDs and names must be unique.")
        return normalized

    def _write(self, records: list[dict[str, Any]]) -> None:
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "templates": records},
            ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise PromptTemplateError("The prompt template store exceeds its 8 MiB limit.")
        self._check_paths(create=True)
        try:
            atomic_write_text(self.path, payload)
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        except OSError:
            raise PromptTemplateError("The prompt template store could not be saved safely.") from None

    def list_page(self, *, limit: int = 50, offset: int = 0, search: str = "") -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise PromptTemplateError("Prompt template page size must be from 1 through 100.")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1_000_000:
            raise PromptTemplateError("Prompt template offset is invalid.")
        if not isinstance(search, str) or len(search) > 120:
            raise PromptTemplateError("Prompt template search is limited to 120 characters.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            records = self._read()
        filtered = [item for item in records if not search or search.casefold() in (item["name"] + " " + item["description"] + " " + item["body"]).casefold()]
        filtered.sort(key=lambda item: (item["updated_at"], item["id"]), reverse=True)
        page = filtered[offset:offset + limit]
        summaries = [{
            "id": item["id"], "name": item["name"], "description": item["description"],
            "version": item["version"], "created_at": item["created_at"], "updated_at": item["updated_at"],
            "variable_count": len(item["variables"]), "content_sha256": item["content_sha256"],
        } for item in page]
        return {"schema_version": SCHEMA_VERSION, "templates": summaries, "total": len(filtered),
                "limit": limit, "offset": offset, "has_more": offset + len(page) < len(filtered)}

    def get(self, template_id: str) -> dict[str, Any] | None:
        if not isinstance(template_id, str) or not _TEMPLATE_ID.fullmatch(template_id):
            raise PromptTemplateError("Prompt template ID is invalid.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            records = self._read()
        item = next((record for record in records if record["id"] == template_id), None)
        if item is None:
            return None
        return {key: value for key, value in item.items() if key != "revisions"}

    def create(self, *, name: str, description: str = "", body: str, variables: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        content = _normalize_content(name, description, body, variables or [])
        now = _now()
        record = {
            "id": uuid.uuid4().hex, "version": 1, "created_at": now, "updated_at": now,
            **content, "content_sha256": _content_digest(content), "revisions": [],
        }
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            records = self._read()
            if len(records) >= MAX_TEMPLATES:
                raise PromptTemplateError("The workspace prompt template limit has been reached.")
            if any(item["name"].casefold() == record["name"].casefold() for item in records):
                raise PromptTemplateError("A prompt template with this name already exists.")
            records.append(record)
            self._write(records)
        return {key: value for key, value in record.items() if key != "revisions"}

    def update(self, template_id: str, *, expected_version: int, name: str, description: str = "", body: str,
               variables: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if not isinstance(template_id, str) or not _TEMPLATE_ID.fullmatch(template_id):
            raise PromptTemplateError("Prompt template ID is invalid.")
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or not 1 <= expected_version <= 1_000_000:
            raise PromptTemplateError("Expected prompt template version is invalid.")
        content = _normalize_content(name, description, body, variables or [])
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            records = self._read()
            item = next((record for record in records if record["id"] == template_id), None)
            if item is None:
                return None
            if item["version"] != expected_version:
                raise PromptTemplateConflict("This template changed since it was loaded. Refresh before saving.")
            if any(record["id"] != template_id and record["name"].casefold() == content["name"].casefold() for record in records):
                raise PromptTemplateError("A prompt template with this name already exists.")
            previous = _snapshot(item)
            revisions = [*item["revisions"], previous][-MAX_REVISIONS:]
            updated = {
                "id": item["id"], "version": item["version"] + 1,
                "created_at": item["created_at"], "updated_at": _now(),
                **content, "content_sha256": _content_digest(content), "revisions": revisions,
            }
            records[records.index(item)] = updated
            self._write(records)
        return {key: value for key, value in updated.items() if key != "revisions"}

    def versions(self, template_id: str) -> list[dict[str, Any]] | None:
        record = self.get_with_revisions(template_id)
        if record is None:
            return None
        items = [{
            "version": item["version"], "updated_at": item["updated_at"], "name": item["name"],
            "content_sha256": item["content_sha256"], "current": False,
        } for item in record["revisions"]]
        items.append({
            "version": record["version"], "updated_at": record["updated_at"], "name": record["name"],
            "content_sha256": record["content_sha256"], "current": True,
        })
        return sorted(items, key=lambda item: item["version"], reverse=True)

    def get_with_revisions(self, template_id: str) -> dict[str, Any] | None:
        if not isinstance(template_id, str) or not _TEMPLATE_ID.fullmatch(template_id):
            raise PromptTemplateError("Prompt template ID is invalid.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            records = self._read()
        return next((record for record in records if record["id"] == template_id), None)

    def get_revision(self, template_id: str, version: int) -> dict[str, Any] | None:
        if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 1_000_000:
            raise PromptTemplateError("Prompt template revision is invalid.")
        record = self.get_with_revisions(template_id)
        if record is None:
            return None
        if version == record["version"]:
            return {key: record[key] for key in ("id", "version", "created_at", "updated_at", *_TEMPLATE_CONTENT_FIELDS, "content_sha256")}
        historical = next((item for item in record["revisions"] if item["version"] == version), None)
        if historical is None:
            raise PromptTemplateError("That prompt template revision is no longer retained.")
        return {"id": record["id"], **historical}

    def render(self, template_id: str, values: dict[str, Any], *, version: int | None = None) -> dict[str, Any] | None:
        record = self.get_revision(template_id, version) if version is not None else self.get(template_id)
        if record is None:
            return None
        rendered = render_prompt(record["body"], record["variables"], values)
        return {
            "template_id": record["id"], "template_name": record["name"],
            "version": record["version"], "rendered": rendered,
            "variables_used": sorted(set(_PLACEHOLDER.findall(record["body"]))),
            "execution_performed": False, "model_invoked": False,
        }

    def delete(self, template_id: str, *, expected_version: int) -> bool:
        if not isinstance(template_id, str) or not _TEMPLATE_ID.fullmatch(template_id):
            raise PromptTemplateError("Prompt template ID is invalid.")
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or not 1 <= expected_version <= 1_000_000:
            raise PromptTemplateError("Expected prompt template version is invalid.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            records = self._read()
            item = next((record for record in records if record["id"] == template_id), None)
            if item is None:
                return False
            if item["version"] != expected_version:
                raise PromptTemplateConflict("This template changed since it was loaded. Refresh before deleting.")
            records.remove(item)
            self._write(records)
        return True
