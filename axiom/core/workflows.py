"""Persistence and validation for non-executable workflow graphs."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock

WORKFLOW_NODE_TYPES = frozenset(
    {"dataset.profile", "model.inspect", "training.plan", "evaluation.run", "report.export"}
)
WORKFLOW_SCHEMA_VERSION = 1
WORKFLOW_VERSION_HISTORY_LIMIT = 10
WORKFLOW_RECORD_LIMIT = 200
_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class WorkflowStoreFullError(ValueError):
    """Raised instead of silently discarding an older workflow design."""


def validate_workflow_graph(
    name: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
) -> None:
    if not name.strip() or len(name) > 120:
        raise ValueError("Workflow name must contain 1 to 120 characters.")
    if not nodes or len(nodes) > 64:
        raise ValueError("A workflow must contain between 1 and 64 nodes.")
    if len(edges) > 128:
        raise ValueError("A workflow can contain at most 128 connections.")

    node_ids: set[str] = set()
    adjacency: dict[str, list[str]] = {}
    incoming_counts: dict[str, int] = {}
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
            raise ValueError("Workflow node IDs must be 1 to 64 letters, digits, underscores, or hyphens.")
        if node_id in node_ids:
            raise ValueError(f"Duplicate workflow node ID: {node_id}")
        if not isinstance(node_type, str) or node_type not in WORKFLOW_NODE_TYPES:
            raise ValueError(f"Unsupported workflow node type: {node_type}")
        node_ids.add(node_id)
        adjacency[node_id] = []
        incoming_counts[node_id] = 0

    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or source not in node_ids
            or target not in node_ids
        ):
            raise ValueError("Every workflow connection must reference an existing node.")
        if source == target:
            raise ValueError("A workflow node cannot connect to itself.")
        key = (source, target)
        if key in seen_edges:
            raise ValueError("Duplicate workflow connection.")
        seen_edges.add(key)
        adjacency[source].append(target)
        incoming_counts[target] += 1

    ready = [node_id for node_id, count in incoming_counts.items() if count == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for target in adjacency[node_id]:
            incoming_counts[target] -= 1
            if incoming_counts[target] == 0:
                ready.append(target)
    if visited != len(node_ids):
        raise ValueError("Workflow connections must be acyclic.")


class WorkflowStore:
    """Store a bounded list of locally authored workflow graphs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".axiom")
        self.records_file = self.root / "workflows.json"
        self.lock_file = self.root / ".workflows.lock"

    def _read(self) -> list[dict[str, Any]]:
        if self.root.is_symlink() or self.records_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid workflow store path: {self.records_file}")
        if not self.records_file.exists():
            return []
        if not self.records_file.is_file() or self.records_file.stat().st_size > 4_000_000:
            raise ValueError(f"Invalid workflow store: {self.records_file}")
        try:
            records = json.loads(self.records_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid workflow store: {self.records_file}") from exc
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError(f"Invalid workflow store: {self.records_file}")
        seen_ids: set[str] = set()
        for record in records:
            workflow_id = record.get("id")
            if not isinstance(workflow_id, str) or not workflow_id or workflow_id in seen_ids:
                raise ValueError(f"Invalid or duplicate workflow ID in {self.records_file}")
            seen_ids.add(workflow_id)
            record.setdefault("schema_version", WORKFLOW_SCHEMA_VERSION)
            record.setdefault("revision", 1)
            record.setdefault("versions", [])
            if (
                type(record["schema_version"]) is not int
                or record["schema_version"] != WORKFLOW_SCHEMA_VERSION
                or type(record["revision"]) is not int
                or record["revision"] < 1
                or not isinstance(record["versions"], list)
                or len(record["versions"]) > WORKFLOW_VERSION_HISTORY_LIMIT
                or any(not isinstance(version, dict) for version in record["versions"])
            ):
                raise ValueError(f"Invalid workflow schema in {self.records_file}")
        return records

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Workflow list limit must be between 1 and 1000.")
        return self._read()[:limit]

    def list_page(
        self,
        *,
        limit: int = 20,
        after_workflow_id: str | None = None,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Workflow page limit must be between 1 and 50.")
        if after_workflow_id is not None and (
            not isinstance(after_workflow_id, str) or not after_workflow_id
        ):
            raise ValueError("Workflow history cursor must be a non-empty workflow ID.")
        records = self._read()
        if len(records) > WORKFLOW_RECORD_LIMIT:
            raise ValueError("Workflow history exceeds the supported record limit.")
        offset = 0
        if after_workflow_id is not None:
            cursor_index = next(
                (index for index, record in enumerate(records) if record.get("id") == after_workflow_id),
                None,
            )
            if cursor_index is None:
                raise KeyError("Workflow history cursor was not found.")
            offset = cursor_index + 1
        return {
            "workflows": records[offset:offset + limit],
            "total": len(records),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(records),
        }

    def get(self, workflow_id: str) -> dict[str, Any] | None:
        return next((item for item in self._read() if item.get("id") == workflow_id), None)

    def save(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("Workflow records must include a string ID.")
        if self.root.is_symlink() or self.records_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid workflow store path: {self.records_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            records = self._read()
            if any(existing.get("id") == record["id"] for existing in records):
                raise ValueError(f"Workflow already exists: {record['id']}")
            if len(records) >= WORKFLOW_RECORD_LIMIT:
                raise WorkflowStoreFullError(
                    f"Workflow history is full ({WORKFLOW_RECORD_LIMIT} designs); remove one before saving."
                )
            stored = {
                **record,
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "revision": 1,
                "versions": [],
            }
            records.insert(0, stored)
            self._write_records(records)

    def update(
        self,
        workflow_id: str,
        *,
        name: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, str]],
        updated_at: str | None = None,
    ) -> dict[str, Any] | None:
        self._check_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            records = self._read()
            for index, current in enumerate(records):
                if current.get("id") != workflow_id:
                    continue
                previous = self._version_snapshot(current)
                record = {
                    **current,
                    "schema_version": WORKFLOW_SCHEMA_VERSION,
                    "revision": current.get("revision", 1) + 1,
                    "versions": [*current.get("versions", []), previous][
                        -WORKFLOW_VERSION_HISTORY_LIMIT:
                    ],
                    "name": name,
                    "updated_at": updated_at or datetime.now(UTC).isoformat(),
                    "nodes": nodes,
                    "edges": edges,
                }
                records[index] = record
                self._write_records(records)
                return record
        return None

    def duplicate(
        self,
        workflow_id: str,
        *,
        new_id: str,
        name: str,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        self._check_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            records = self._read()
            source = next((record for record in records if record.get("id") == workflow_id), None)
            if source is None:
                return None
            if len(records) >= WORKFLOW_RECORD_LIMIT:
                raise WorkflowStoreFullError(
                    f"Workflow history is full ({WORKFLOW_RECORD_LIMIT} designs); remove one before duplicating."
                )
            timestamp = now or datetime.now(UTC).isoformat()
            duplicate = {
                **source,
                "id": new_id,
                "name": name,
                "created_at": timestamp,
                "updated_at": timestamp,
                "revision": 1,
                "versions": [],
                "nodes": [dict(node) for node in source.get("nodes", [])],
                "edges": [dict(edge) for edge in source.get("edges", [])],
            }
            records.insert(0, duplicate)
            self._write_records(records)
            return duplicate

    def versions(self, workflow_id: str) -> list[dict[str, Any]] | None:
        record = self.get(workflow_id)
        if record is None:
            return None
        return list(record.get("versions", []))

    def _check_paths(self) -> None:
        if self.root.is_symlink() or self.records_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid workflow store path: {self.records_file}")

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        serialized = json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > 4_000_000:
            raise ValueError("Workflow history exceeds the 4 MB workspace limit.")
        atomic_write_text(self.records_file, serialized + "\n")

    @staticmethod
    def _version_snapshot(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "revision": record.get("revision", 1),
            "updated_at": record.get("updated_at", record.get("created_at", "")),
            "name": record.get("name", ""),
            "nodes": [dict(node) for node in record.get("nodes", [])],
            "edges": [dict(edge) for edge in record.get("edges", [])],
        }
