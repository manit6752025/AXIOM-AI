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
WORKFLOW_SCHEMA_VERSION = 2
WORKFLOW_LEGACY_SCHEMA_VERSION = 1
WORKFLOW_VERSION_HISTORY_LIMIT = 10
WORKFLOW_RECORD_LIMIT = 200
_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _port(port_id: str, data_type: str, *, required: bool = False) -> dict[str, Any]:
    return {"id": port_id, "data_type": data_type, "required": required}


WORKFLOW_NODE_PORT_SCHEMAS: dict[str, dict[str, Any]] = {
    "dataset.profile": {
        "inputs": [_port("artifact", "artifact")],
        "outputs": [_port("dataset", "dataset"), _port("artifact", "artifact")],
        "required_any_inputs": [],
    },
    "model.inspect": {
        "inputs": [_port("artifact", "artifact")],
        "outputs": [_port("model", "model"), _port("artifact", "artifact")],
        "required_any_inputs": [],
    },
    "training.plan": {
        "inputs": [
            _port("dataset", "dataset", required=True),
            _port("model", "model", required=True),
            _port("artifact", "artifact"),
        ],
        "outputs": [_port("plan", "plan"), _port("artifact", "artifact")],
        "required_any_inputs": [],
    },
    "evaluation.run": {
        "inputs": [
            _port("plan", "plan", required=True),
            _port("dataset", "dataset"),
            _port("artifact", "artifact"),
        ],
        "outputs": [_port("evaluation", "evaluation"), _port("artifact", "artifact")],
        "required_any_inputs": [],
    },
    "report.export": {
        "inputs": [
            _port("dataset", "dataset"),
            _port("plan", "plan"),
            _port("evaluation", "evaluation"),
            _port("artifact", "artifact"),
        ],
        "outputs": [_port("report", "report"), _port("artifact", "artifact")],
        "required_any_inputs": ["dataset", "plan", "evaluation"],
    },
}


def workflow_node_port_schema(node_type: str) -> dict[str, Any]:
    try:
        schema = WORKFLOW_NODE_PORT_SCHEMAS[node_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow node type: {node_type}") from exc
    return json.loads(json.dumps(schema))


def normalize_workflow_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Attach authoritative ports and upgrade untyped legacy edges deterministically."""
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 64:
        raise ValueError("A workflow must contain between 1 and 64 nodes.")
    if not isinstance(edges, list) or len(edges) > 128:
        raise ValueError("A workflow can contain at most 128 connections.")

    normalized_nodes: list[dict[str, Any]] = []
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Workflow nodes must be objects.")
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
            raise ValueError("Workflow node IDs must be 1 to 64 letters, digits, underscores, or hyphens.")
        if node_id in nodes_by_id:
            raise ValueError(f"Duplicate workflow node ID: {node_id}")
        schema = workflow_node_port_schema(node_type) if isinstance(node_type, str) else None
        if schema is None:
            raise ValueError(f"Unsupported workflow node type: {node_type}")
        supplied_ports = node.get("ports")
        if supplied_ports is not None and supplied_ports != schema:
            raise ValueError(f"Workflow node {node_id} has an invalid port schema.")
        normalized = {**node, "ports": schema}
        normalized_nodes.append(normalized)
        nodes_by_id[node_id] = normalized

    normalized_edges: list[dict[str, str]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Workflow connections must be objects.")
        source = edge.get("source")
        target = edge.get("target")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or source not in nodes_by_id
            or target not in nodes_by_id
        ):
            raise ValueError("Every workflow connection must reference an existing node.")

        source_ports = nodes_by_id[source]["ports"]["outputs"]
        target_ports = nodes_by_id[target]["ports"]["inputs"]
        source_port = edge.get("source_port")
        target_port = edge.get("target_port")
        if source_port is None and target_port is None:
            compatible = [
                (out_port, in_port)
                for out_port in source_ports
                for in_port in target_ports
                if out_port["data_type"] == in_port["data_type"]
                and out_port["data_type"] != "artifact"
            ]
            if compatible:
                source_port, target_port = compatible[0][0]["id"], compatible[0][1]["id"]
            else:
                # Schema-v1 edges had no port identity. Preserve their topology
                # through the explicit generic-artifact compatibility port.
                source_port, target_port = "artifact", "artifact"
        elif source_port is None or target_port is None:
            raise ValueError("Workflow connections must specify both source_port and target_port.")

        if not isinstance(source_port, str) or not isinstance(target_port, str):
            raise ValueError("Workflow connection port IDs must be strings.")
        normalized_edges.append({
            "source": source,
            "target": target,
            "source_port": source_port,
            "target_port": target_port,
        })
    return normalized_nodes, normalized_edges


class WorkflowStoreFullError(ValueError):
    """Raised instead of silently discarding an older workflow design."""


def validate_workflow_graph(
    name: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise ValueError("Workflow name must contain 1 to 120 characters.")
    normalized_nodes, normalized_edges = normalize_workflow_graph(nodes, edges)

    node_ids = {node["id"] for node in normalized_nodes}
    nodes_by_id = {node["id"]: node for node in normalized_nodes}
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    incoming_counts: dict[str, int] = {node_id: 0 for node_id in node_ids}
    connected_inputs: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    seen_edges: set[tuple[str, str, str, str]] = set()

    for edge in normalized_edges:
        source = edge["source"]
        target = edge["target"]
        source_node = nodes_by_id[source]
        target_node = nodes_by_id[target]
        source_port = next(
            (port for port in source_node["ports"]["outputs"] if port["id"] == edge["source_port"]),
            None,
        )
        target_port = next(
            (port for port in target_node["ports"]["inputs"] if port["id"] == edge["target_port"]),
            None,
        )
        if source_port is None or target_port is None:
            raise ValueError("Workflow connection references a port that is not defined by its node.")
        if source_port["data_type"] != target_port["data_type"]:
            raise ValueError("Workflow connection ports have incompatible data types.")
        if source == target:
            raise ValueError("A workflow node cannot connect to itself.")
        key = (source, edge["source_port"], target, edge["target_port"])
        if key in seen_edges:
            raise ValueError("Duplicate workflow connection.")
        seen_edges.add(key)
        if edge["target_port"] in connected_inputs[target]:
            raise ValueError(f"Workflow input port {target}.{edge['target_port']} has more than one connection.")
        connected_inputs[target].add(edge["target_port"])
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

    missing_inputs: list[dict[str, str]] = []
    for node in normalized_nodes:
        node_id = node["id"]
        schema = node["ports"]
        for port in schema["inputs"]:
            if port["required"] and port["id"] not in connected_inputs[node_id]:
                missing_inputs.append({
                    "node_id": node_id,
                    "port_id": port["id"],
                    "reason": "Required input is not connected.",
                })
        required_any = schema.get("required_any_inputs", [])
        if required_any and not any(port_id in connected_inputs[node_id] for port_id in required_any):
            missing_inputs.append({
                "node_id": node_id,
                "port_id": "one_of:" + "|".join(required_any),
                "reason": "Connect at least one input to this node.",
            })
    return missing_inputs


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
            record.setdefault("schema_version", WORKFLOW_LEGACY_SCHEMA_VERSION)
            record.setdefault("revision", 1)
            record.setdefault("versions", [])
            if (
                type(record["schema_version"]) is not int
                or record["schema_version"] not in {WORKFLOW_LEGACY_SCHEMA_VERSION, WORKFLOW_SCHEMA_VERSION}
                or type(record["revision"]) is not int
                or record["revision"] < 1
                or not isinstance(record["versions"], list)
                or len(record["versions"]) > WORKFLOW_VERSION_HISTORY_LIMIT
                or any(not isinstance(version, dict) for version in record["versions"])
            ):
                raise ValueError(f"Invalid workflow schema in {self.records_file}")
            try:
                record["nodes"], record["edges"] = normalize_workflow_graph(
                    record.get("nodes"), record.get("edges"),
                )
                for version in record["versions"]:
                    version["nodes"], version["edges"] = normalize_workflow_graph(
                        version.get("nodes"), version.get("edges"),
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid workflow graph in {self.records_file}") from exc
            record["schema_version"] = WORKFLOW_SCHEMA_VERSION
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
        search: str | None = None,
    ) -> dict[str, Any]:
        """Return a stable cursor page, applying search before calculating offsets."""
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Workflow page limit must be between 1 and 50.")
        if after_workflow_id is not None and (
            not isinstance(after_workflow_id, str) or not after_workflow_id
        ):
            raise ValueError("Workflow history cursor must be a non-empty workflow ID.")
        if search is not None and (not isinstance(search, str) or len(search) > 120):
            raise ValueError("Workflow search must contain at most 120 characters.")
        records = self._read()
        if len(records) > WORKFLOW_RECORD_LIMIT:
            raise ValueError("Workflow history exceeds the supported record limit.")
        needle = (search or "").strip().casefold()
        if needle:
            def matches(record: dict[str, Any]) -> bool:
                name = record.get("name")
                workflow_id = record.get("id")
                node_types = [
                    node.get("type")
                    for node in record.get("nodes", [])
                    if isinstance(node, dict)
                ]
                return any(
                    needle in value.casefold()
                    for value in (name, workflow_id, *node_types)
                    if isinstance(value, str)
                )
            records = [record for record in records if matches(record)]
        offset = 0
        if after_workflow_id is not None:
            cursor_index = next(
                (index for index, record in enumerate(records) if record.get("id") == after_workflow_id),
                None,
            )
            if cursor_index is None:
                raise KeyError("Workflow history cursor was not found for the selected filter.")
            offset = cursor_index + 1
        page = records[offset:offset + limit]
        has_more = offset + len(page) < len(records)
        return {
            "workflows": page,
            "total": len(records),
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_cursor": page[-1]["id"] if has_more and page else None,
            "filters": {"search": needle},
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
            normalized_nodes, normalized_edges = normalize_workflow_graph(
                record.get("nodes"), record.get("edges"),
            )
            stored = {
                **record,
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "revision": 1,
                "versions": [],
                "nodes": normalized_nodes,
                "edges": normalized_edges,
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
        normalized_nodes, normalized_edges = normalize_workflow_graph(nodes, edges)
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
                    "nodes": normalized_nodes,
                    "edges": normalized_edges,
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
