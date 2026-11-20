"""Append-only, hash-chained local security event journal.

Only predefined event types and static summaries are accepted. Review labels
are explicitly self-attested and are not authenticated account identities.
The journal is workspace-local, bounded, atomically replaced, and never cleared
or silently rotated when full.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from datetime import UTC, datetime
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import active_workspace_root
from axiom.path_safety import SafePathError, open_regular_file

LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
MAX_EVENTS = 2_000
MAX_STORE_BYTES = 5 * 1024 * 1024
MAX_PAGE_SIZE = 100
_GENESIS_HASH = "0" * 64
_EVENT_ID = re.compile(r"^[a-f0-9]{32}$")
_REVIEWER_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{1,47}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_REDACTED_VALUE = "[REDACTED]"
_TOOL_DECISION_EVENT = "mcp.tool.decision"
_TOOL_DECISION_REASONS = {"explicit_grant", "not_granted", "unknown_tool", "invalid_policy"}

_EVENT_SUMMARIES = {
    "mcp.profile.created": "Created a default-deny MCP profile",
    "mcp.permissions.saved": "Updated an MCP tool permission profile",
    "mcp.profile.activated": "Changed the active MCP permission profile",
    _TOOL_DECISION_EVENT: "Recorded a redacted MCP tool permission decision",
}
_REVIEW_EVENT = "security.event.reviewed"
_REVIEW_SUMMARY = "Recorded an administrator review acknowledgement"


class SecurityEventError(ValueError):
    """The security event journal is invalid, unsafe, or at capacity."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecurityEventError("Security event data is not canonical JSON.") from exc


def _event_digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "hash"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SecurityEventStore:
    """Maintain one bounded append-only event chain for a workspace."""

    def __init__(self, root: Path | None = None) -> None:
        supplied = Path(root) if root is not None else active_workspace_root()
        if supplied.is_symlink():
            raise SecurityEventError("Workspace root must not be a symbolic link.")
        try:
            self.root = supplied.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SecurityEventError("Workspace root is unavailable.") from exc
        if not self.root.is_dir():
            raise SecurityEventError("Workspace root must be a directory.")
        self._loaded_schema_version = SCHEMA_VERSION
        self.directory = self.root / ".axiom"
        self.path = self.directory / "security-events.json"
        self.lock_path = self.directory / "security-events.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._check_paths(create=True)
        try:
            if (
                os.name == "posix"
                and getattr(os, "O_NOFOLLOW", 0)
                and getattr(os, "O_DIRECTORY", 0)
                and os.open in getattr(os, "supports_dir_fd", set())
            ):
                directory_fd = os.open(
                    self.directory,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                handle = None
                descriptor = None
                import fcntl

                try:
                    descriptor = os.open(
                        self.lock_path.name,
                        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                        dir_fd=directory_fd,
                    )
                    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                        raise SecurityEventError("Security journal lock must be a regular file.")
                    os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
                    handle = os.fdopen(descriptor, "r+b")
                    descriptor = None
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    self._check_paths()
                    yield
                finally:
                    if handle is not None:
                        try:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                        finally:
                            handle.close()
                    if descriptor is not None:
                        os.close(descriptor)
                    os.close(directory_fd)
            else:
                # Windows uses the portable path-based lock fallback; symlink
                # checks are repeated after acquiring the handle.
                with file_lock(self.lock_path):
                    self._check_paths()
                    yield
        except SecurityEventError:
            raise
        except (OSError, TimeoutError) as exc:
            raise SecurityEventError("Security journal lock could not be acquired safely.") from exc

    def _check_paths(self, *, create: bool = False) -> None:
        if self.directory.is_symlink():
            raise SecurityEventError("Workspace .axiom directory must not be a symbolic link.")
        if self.directory.exists() and not self.directory.is_dir():
            raise SecurityEventError("Workspace .axiom path is not a directory.")
        if create:
            self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        if self.directory.exists():
            try:
                resolved = self.directory.resolve(strict=True)
                resolved.relative_to(self.root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise SecurityEventError("Security journal directory escapes the workspace.") from exc
            if os.name == "posix" and self.directory.stat().st_mode & 0o022:
                raise SecurityEventError("Security journal directory is writable by group or other users.")
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise SecurityEventError("Security journal and lock paths must not be symbolic links.")

    def _read_locked(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.path.exists():
            return []
        try:
            with open_regular_file(self.path) as handle:
                info = os.fstat(handle.fileno())
                if info.st_size <= 0 or info.st_size > MAX_STORE_BYTES:
                    raise SecurityEventError("Security event journal is empty or exceeds its 5 MiB limit.")
                if os.name == "posix" and info.st_mode & 0o077:
                    raise SecurityEventError("Security event journal permissions are broader than owner-only.")
                raw_bytes = handle.read(MAX_STORE_BYTES + 1)
            if len(raw_bytes) > MAX_STORE_BYTES:
                raise SecurityEventError("Security event journal exceeds its 5 MiB limit.")
            document = json.loads(raw_bytes.decode("utf-8"))
        except SecurityEventError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, SafePathError) as exc:
            raise SecurityEventError("Security event journal could not be read safely.") from exc

        schema_version = document.get("schema_version") if isinstance(document, dict) else None
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
        ):
            raise SecurityEventError("Unsupported security event journal schema.")
        self._loaded_schema_version = schema_version
        events = document.get("events")
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise SecurityEventError("Security event journal has an invalid event collection.")
        previous = _GENESIS_HASH
        seen_ids: set[str] = set()
        for expected_seq, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                raise SecurityEventError("Security event journal contains an invalid record.")
            required = {
                "seq", "event_id", "occurred_at", "event_type", "summary",
                "prev_hash", "hash",
            }
            decision_fields = {"tool_name", "decision", "profile_name", "reason", "arguments"}
            allowed = required | {"reviewer_label", "target_event_id"} | decision_fields
            if not required.issubset(event) or not set(event).issubset(allowed):
                raise SecurityEventError("Security event journal record fields are invalid.")
            sequence = event.get("seq")
            if isinstance(sequence, bool) or sequence != expected_seq:
                raise SecurityEventError("Security event sequence is invalid.")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id) or event_id in seen_ids:
                raise SecurityEventError("Security event IDs are invalid or duplicated.")
            occurred_at = event.get("occurred_at")
            if not isinstance(occurred_at, str) or len(occurred_at) > 40:
                raise SecurityEventError("Security event timestamp is invalid.")
            try:
                parsed_timestamp = datetime.fromisoformat(occurred_at)
            except ValueError as exc:
                raise SecurityEventError("Security event timestamp is invalid.") from exc
            if parsed_timestamp.tzinfo is None:
                raise SecurityEventError("Security event timestamp must include a timezone.")
            event_type = event.get("event_type")
            summary = event.get("summary")
            if not isinstance(event_type, str) or not isinstance(summary, str):
                raise SecurityEventError("Security event type and summary must be text.")
            previous_hash = event.get("prev_hash")
            current_hash = event.get("hash")
            if (
                not isinstance(previous_hash, str)
                or not _SHA256.fullmatch(previous_hash)
                or not isinstance(current_hash, str)
                or not _SHA256.fullmatch(current_hash)
                or previous_hash != previous
                or current_hash != _event_digest(event)
            ):
                raise SecurityEventError("Security event hash chain verification failed.")
            if event_type == _REVIEW_EVENT:
                if set(event) != required | {"reviewer_label", "target_event_id"} or summary != _REVIEW_SUMMARY:
                    raise SecurityEventError("Security review record is invalid.")
                target = event.get("target_event_id")
                reviewer = event.get("reviewer_label")
                if not isinstance(target, str) or not _EVENT_ID.fullmatch(target):
                    raise SecurityEventError("Security review target is invalid.")
                if not isinstance(reviewer, str) or not _REVIEWER_LABEL.fullmatch(reviewer):
                    raise SecurityEventError("Security reviewer label is invalid.")
            elif event_type == _TOOL_DECISION_EVENT:
                if document.get("schema_version") != SCHEMA_VERSION or set(event) != required | decision_fields:
                    raise SecurityEventError("MCP tool decision record fields are invalid.")
                tool_name = event.get("tool_name")
                decision = event.get("decision")
                profile_name = event.get("profile_name")
                reason = event.get("reason")
                arguments = event.get("arguments")
                if not isinstance(tool_name, str) or not _TOOL_NAME.fullmatch(tool_name):
                    raise SecurityEventError("MCP tool decision name is invalid.")
                if not isinstance(decision, str) or decision not in {"allowed", "denied"}:
                    raise SecurityEventError("MCP tool decision outcome is invalid.")
                if not isinstance(profile_name, str) or not (
                    _PROFILE_NAME.fullmatch(profile_name) or profile_name == "unavailable"
                ):
                    raise SecurityEventError("MCP tool decision profile is invalid.")
                if not isinstance(reason, str) or reason not in _TOOL_DECISION_REASONS:
                    raise SecurityEventError("MCP tool decision reason is invalid.")
                if (
                    not isinstance(arguments, dict)
                    or len(arguments) > 64
                    or any(
                        not isinstance(name, str)
                        or not _ARGUMENT_NAME.fullmatch(name)
                        or value != _REDACTED_VALUE
                        for name, value in arguments.items()
                    )
                    or summary != _EVENT_SUMMARIES[_TOOL_DECISION_EVENT]
                ):
                    raise SecurityEventError("MCP tool decision arguments are not safely redacted.")
            else:
                if set(event) != required or event_type not in _EVENT_SUMMARIES or summary != _EVENT_SUMMARIES.get(event_type):
                    raise SecurityEventError("Security event type or summary is not allowlisted.")
            seen_ids.add(event_id)
            previous = event["hash"]
        targets = {event["event_id"] for event in events if event["event_type"] != _REVIEW_EVENT}
        if any(event.get("target_event_id") not in targets for event in events if event["event_type"] == _REVIEW_EVENT):
            raise SecurityEventError("Security review references an unknown event.")
        return events

    def _write_locked(self, events: list[dict[str, Any]]) -> None:
        self._check_paths(create=True)
        document = {"schema_version": SCHEMA_VERSION, "events": events}
        payload = json.dumps(
            document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise SecurityEventError("Security event journal would exceed its 5 MiB limit.")
        atomic_write_text(self.path, payload)
        self._loaded_schema_version = SCHEMA_VERSION
        if os.name == "posix":
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)

    def _append_locked(
        self,
        events: list[dict[str, Any]],
        event_type: str,
        *,
        reviewer_label: str | None = None,
        target_event_id: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(events) >= MAX_EVENTS:
            raise SecurityEventError("Security event journal is full; export and archive it before further events.")
        record: dict[str, Any] = {
            "seq": len(events) + 1,
            "event_id": uuid.uuid4().hex,
            "occurred_at": _utc_now(),
            "event_type": event_type,
            "summary": _REVIEW_SUMMARY if event_type == _REVIEW_EVENT else _EVENT_SUMMARIES[event_type],
            "prev_hash": events[-1]["hash"] if events else _GENESIS_HASH,
        }
        if reviewer_label is not None:
            record["reviewer_label"] = reviewer_label
        if target_event_id is not None:
            record["target_event_id"] = target_event_id
        if extra_fields is not None:
            record.update(extra_fields)
        record["hash"] = _event_digest(record)
        events.append(record)
        self._write_locked(events)
        return dict(record)

    def record(self, event_type: str) -> dict[str, Any]:
        """Append one allowlisted security event; no user-provided text is accepted."""
        if event_type not in _EVENT_SUMMARIES or event_type == _TOOL_DECISION_EVENT:
            raise SecurityEventError("Security event type is not allowlisted for generic recording.")
        self._check_paths(create=True)
        try:
            with file_lock(self.lock_path):
                self._check_paths()
                events = self._read_locked()
                return self._append_locked(events, event_type)
        except (OSError, TimeoutError) as exc:
            raise SecurityEventError("Security event could not be recorded safely.") from exc

    def record_tool_decision(
        self,
        tool_name: str,
        profile_name: str,
        allowed: bool,
        argument_names: list[str] | tuple[str, ...],
        reason: str,
    ) -> dict[str, Any]:
        """Record an allow/deny decision using argument names only, never their values."""
        if not isinstance(tool_name, str) or not _TOOL_NAME.fullmatch(tool_name):
            raise SecurityEventError("MCP tool decision name is invalid.")
        if not isinstance(profile_name, str) or not (
            _PROFILE_NAME.fullmatch(profile_name) or profile_name == "unavailable"
        ):
            raise SecurityEventError("MCP tool decision profile is invalid.")
        if type(allowed) is not bool:
            raise SecurityEventError("MCP tool decision outcome must be boolean.")
        if not isinstance(reason, str) or reason not in _TOOL_DECISION_REASONS:
            raise SecurityEventError("MCP tool decision reason is invalid.")
        if (
            not isinstance(argument_names, (list, tuple))
            or len(argument_names) > 64
            or any(
                not isinstance(name, str) or not _ARGUMENT_NAME.fullmatch(name)
                for name in argument_names
            )
            or len(set(argument_names)) != len(argument_names)
        ):
            raise SecurityEventError("MCP tool argument names are invalid.")
        redacted_arguments = {name: _REDACTED_VALUE for name in argument_names}
        self._check_paths(create=True)
        try:
            with self._locked():
                events = self._read_locked()
                return self._append_locked(
                    events,
                    _TOOL_DECISION_EVENT,
                    extra_fields={
                        "tool_name": tool_name,
                        "decision": "allowed" if allowed else "denied",
                        "profile_name": profile_name,
                        "reason": reason,
                        "arguments": redacted_arguments,
                    },
                )
        except SecurityEventError:
            raise
        except (OSError, TimeoutError) as exc:
            raise SecurityEventError("MCP tool decision could not be audited safely.") from exc

    def review(self, event_id: str, reviewer_label: str) -> dict[str, Any]:
        """Append a review acknowledgement; the reviewer label is self-attested."""
        if not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id):
            raise SecurityEventError("Security event ID is invalid.")
        if not isinstance(reviewer_label, str):
            raise SecurityEventError("Reviewer label must be text.")
        label = reviewer_label.strip()
        if not _REVIEWER_LABEL.fullmatch(label):
            raise SecurityEventError("Reviewer label must be 2–48 safe characters.")
        self._check_paths(create=True)
        try:
            with self._locked():
                self._check_paths()
                events = self._read_locked()
                target = next((item for item in events if item["event_id"] == event_id), None)
                if target is None or target["event_type"] == _REVIEW_EVENT:
                    raise SecurityEventError("Security event was not found.")
                existing = next(
                    (item for item in reversed(events)
                     if item["event_type"] == _REVIEW_EVENT and item["target_event_id"] == event_id),
                    None,
                )
                if existing is not None:
                    return {"review": dict(existing), "already_reviewed": True}
                entry = self._append_locked(
                    events, _REVIEW_EVENT, reviewer_label=label, target_event_id=event_id,
                )
                return {"review": entry, "already_reviewed": False}
        except SecurityEventError:
            raise
        except (OSError, TimeoutError) as exc:
            raise SecurityEventError("Security review could not be recorded safely.") from exc

    def verify(self) -> dict[str, Any]:
        """Verify the complete retained chain without changing it."""
        self._check_paths()
        if not self.path.exists():
            return {
                "schema_version": SCHEMA_VERSION, "valid": True, "event_count": 0,
                "max_events": MAX_EVENTS, "head_hash": _GENESIS_HASH, "capacity_reached": False,
            }
        with self._locked():
            events = self._read_locked()
        return {
            "schema_version": self._loaded_schema_version,
            "valid": True,
            "event_count": len(events),
            "max_events": MAX_EVENTS,
            "head_hash": events[-1]["hash"] if events else _GENESIS_HASH,
            "capacity_reached": len(events) >= MAX_EVENTS,
        }

    def list_page(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
            raise SecurityEventError(f"Limit must be between 1 and {MAX_PAGE_SIZE}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < MAX_EVENTS:
            raise SecurityEventError(f"Offset must be between 0 and {MAX_EVENTS - 1}.")
        self._check_paths()
        if self.path.exists():
            with self._locked():
                events = self._read_locked()
        else:
            events = []
        reviews = {
            event["target_event_id"]: event
            for event in events if event["event_type"] == _REVIEW_EVENT
        }
        primary = [event for event in events if event["event_type"] != _REVIEW_EVENT]
        total = len(primary)
        page = list(reversed(primary))[offset:offset + limit]
        projected = []
        for event in page:
            review = reviews.get(event["event_id"])
            projected.append({
                **event,
                "reviewed": review is not None,
                "reviewed_at": review["occurred_at"] if review else None,
                "reviewer_label": review.get("reviewer_label") if review else None,
                "reviewer_identity_verified": False if review else None,
            })
        integrity = {
            "schema_version": self._loaded_schema_version,
            "valid": True,
            "event_count": len(events),
            "max_events": MAX_EVENTS,
            "head_hash": events[-1]["hash"] if events else _GENESIS_HASH,
            "capacity_reached": len(events) >= MAX_EVENTS,
        }
        return {
            "schema_version": self._loaded_schema_version,
            "events": projected,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(page) < total,
            "integrity": integrity,
            "privacy": {
                "event_summaries_allowlisted": True,
                "tool_decisions_audited": True,
                "argument_values_stored": False,
                "credential_values_stored": False,
                "arbitrary_user_text_stored": False,
                "reviewer_identity_authenticated": False,
                "journal_is_append_only": True,
            },
        }

    def export(self) -> dict[str, Any]:
        self._check_paths()
        if self.path.exists():
            with self._locked():
                events = self._read_locked()
        else:
            events = []
        return {
            "schema_version": self._loaded_schema_version,
            "exported_at": _utc_now(),
            "integrity": {
                "valid": True,
                "event_count": len(events),
                "max_events": MAX_EVENTS,
                "head_hash": events[-1]["hash"] if events else _GENESIS_HASH,
                "capacity_reached": len(events) >= MAX_EVENTS,
            },
            "events": events,
            "privacy": {
                "credential_values_stored": False,
                "arbitrary_user_text_stored": False,
                "reviewer_identity_authenticated": False,
                "tool_decisions_audited": True,
                "argument_values_stored": False,
            },
        }


def record_security_event(event_type: str) -> dict[str, Any]:
    """Convenience entry point for request-scoped API and CLI actions."""
    return SecurityEventStore().record(event_type)
