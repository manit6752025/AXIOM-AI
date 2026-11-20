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

SCHEMA_VERSION = 1
MAX_EVENTS = 2_000
MAX_STORE_BYTES = 5 * 1024 * 1024
MAX_PAGE_SIZE = 100
_GENESIS_HASH = "0" * 64
_EVENT_ID = re.compile(r"^[a-f0-9]{32}$")
_REVIEWER_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{1,47}$")

_EVENT_SUMMARIES = {
    "mcp.profile.created": "Created a default-deny MCP profile",
    "mcp.permissions.saved": "Updated an MCP tool permission profile",
    "mcp.profile.activated": "Changed the active MCP permission profile",
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
        self.directory = self.root / ".axiom"
        self.path = self.directory / "security-events.json"
        self.lock_path = self.directory / "security-events.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._check_paths(create=True)
        try:
            with file_lock(self.lock_path):
                self._check_paths()
                if os.name == "posix":
                    os.chmod(self.lock_path, stat.S_IRUSR | stat.S_IWUSR)
                yield
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

        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            raise SecurityEventError("Unsupported security event journal schema.")
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
            allowed = required | {"reviewer_label", "target_event_id"}
            if not required.issubset(event) or not set(event).issubset(allowed):
                raise SecurityEventError("Security event journal record fields are invalid.")
            if event.get("seq") != expected_seq or not isinstance(event.get("event_id"), str):
                raise SecurityEventError("Security event sequence is invalid.")
            event_id = event["event_id"]
            if not _EVENT_ID.fullmatch(event_id) or event_id in seen_ids:
                raise SecurityEventError("Security event IDs are invalid or duplicated.")
            if event.get("prev_hash") != previous or event.get("hash") != _event_digest(event):
                raise SecurityEventError("Security event hash chain verification failed.")
            if event.get("event_type") == _REVIEW_EVENT:
                if event.get("summary") != _REVIEW_SUMMARY:
                    raise SecurityEventError("Security review record is invalid.")
                target = event.get("target_event_id")
                reviewer = event.get("reviewer_label")
                if not isinstance(target, str) or not _EVENT_ID.fullmatch(target):
                    raise SecurityEventError("Security review target is invalid.")
                if not isinstance(reviewer, str) or not _REVIEWER_LABEL.fullmatch(reviewer):
                    raise SecurityEventError("Security reviewer label is invalid.")
            elif event.get("event_type") not in _EVENT_SUMMARIES or event.get("summary") != _EVENT_SUMMARIES.get(event.get("event_type")):
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
        if os.name == "posix":
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)

    def _append_locked(
        self,
        events: list[dict[str, Any]],
        event_type: str,
        *,
        reviewer_label: str | None = None,
        target_event_id: str | None = None,
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
        record["hash"] = _event_digest(record)
        events.append(record)
        self._write_locked(events)
        return dict(record)

    def record(self, event_type: str) -> dict[str, Any]:
        """Append one allowlisted security event; no user-provided text is accepted."""
        if event_type not in _EVENT_SUMMARIES:
            raise SecurityEventError("Security event type is not allowlisted.")
        self._check_paths(create=True)
        try:
            with file_lock(self.lock_path):
                self._check_paths()
                events = self._read_locked()
                return self._append_locked(events, event_type)
        except (OSError, TimeoutError) as exc:
            raise SecurityEventError("Security event could not be recorded safely.") from exc

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
            "schema_version": SCHEMA_VERSION,
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
            "schema_version": SCHEMA_VERSION,
            "valid": True,
            "event_count": len(events),
            "max_events": MAX_EVENTS,
            "head_hash": events[-1]["hash"] if events else _GENESIS_HASH,
            "capacity_reached": len(events) >= MAX_EVENTS,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "events": projected,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(page) < total,
            "integrity": integrity,
            "privacy": {
                "event_summaries_allowlisted": True,
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
            "schema_version": SCHEMA_VERSION,
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
            },
        }


def record_security_event(event_type: str) -> dict[str, Any]:
    """Convenience entry point for request-scoped API and CLI actions."""
    return SecurityEventStore().record(event_type)
