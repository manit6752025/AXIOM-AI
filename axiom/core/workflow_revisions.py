"""Bounded structural comparisons between retained workflow revisions."""

from __future__ import annotations

from typing import Any

from axiom.core.workflows import (
    WORKFLOW_SCHEMA_VERSION,
    WORKFLOW_VERSION_HISTORY_LIMIT,
    validate_workflow_graph,
)


class WorkflowRevisionError(ValueError):
    """A stable, safe failure reason for workflow revision comparisons."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _valid_display_text(value: Any, *, limit: int) -> bool:
    if not isinstance(value, str) or not value or len(value) > limit:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not any(ord(char) < 32 or ord(char) == 127 for char in value)


def _validated_graph(
    *,
    name: Any,
    nodes: Any,
    edges: Any,
    revision: Any,
) -> dict[str, Any]:
    if type(revision) is not int or not 1 <= revision <= 1_000_000:
        raise WorkflowRevisionError("workflow_history_invalid")
    if not _valid_display_text(name, limit=120):
        raise WorkflowRevisionError("workflow_history_invalid")
    if type(nodes) is not list or not 1 <= len(nodes) <= 64:
        raise WorkflowRevisionError("workflow_history_invalid")
    if type(edges) is not list or len(edges) > 128:
        raise WorkflowRevisionError("workflow_history_invalid")

    safe_nodes: list[dict[str, str]] = []
    for node in nodes:
        if (
            not isinstance(node, dict)
            or set(node) != {"id", "type"}
            or not isinstance(node.get("id"), str)
            or len(node["id"]) > 64
            or not isinstance(node.get("type"), str)
            or len(node["type"]) > 40
        ):
            raise WorkflowRevisionError("workflow_history_invalid")
        safe_nodes.append({"id": node["id"], "type": node["type"]})

    safe_edges: list[dict[str, str]] = []
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or set(edge) != {"source", "target"}
            or not isinstance(edge.get("source"), str)
            or len(edge["source"]) > 64
            or not isinstance(edge.get("target"), str)
            or len(edge["target"]) > 64
        ):
            raise WorkflowRevisionError("workflow_history_invalid")
        safe_edges.append({"source": edge["source"], "target": edge["target"]})

    try:
        validate_workflow_graph(name, safe_nodes, safe_edges)
    except (AttributeError, TypeError, ValueError) as exc:
        raise WorkflowRevisionError("workflow_history_invalid") from exc

    return {
        "revision": revision,
        "name": name,
        "nodes": safe_nodes,
        "edges": safe_edges,
    }


def compare_workflow_revision(
    workflow: dict[str, Any],
    *,
    against_revision: int,
) -> dict[str, Any]:
    """Compare one retained design snapshot with the current saved graph."""
    if (
        not isinstance(workflow, dict)
        or type(workflow.get("schema_version")) is not int
        or workflow.get("schema_version") != WORKFLOW_SCHEMA_VERSION
    ):
        raise WorkflowRevisionError("workflow_history_invalid")
    if type(against_revision) is not int or not 1 <= against_revision <= 1_000_000:
        raise WorkflowRevisionError("revision_invalid")

    current = _validated_graph(
        name=workflow.get("name"),
        nodes=workflow.get("nodes"),
        edges=workflow.get("edges"),
        revision=workflow.get("revision"),
    )
    current_revision = current["revision"]
    if against_revision > current_revision:
        raise WorkflowRevisionError("revision_not_found")

    if against_revision == current_revision:
        baseline = current
    else:
        raw_versions = workflow.get("versions", [])
        if (
            type(raw_versions) is not list
            or len(raw_versions) > WORKFLOW_VERSION_HISTORY_LIMIT
            or len(raw_versions) != min(current_revision - 1, WORKFLOW_VERSION_HISTORY_LIMIT)
        ):
            raise WorkflowRevisionError("workflow_history_invalid")

        versions: list[dict[str, Any]] = []
        for item in raw_versions:
            if (
                not isinstance(item, dict)
                or set(item) != {"revision", "updated_at", "name", "nodes", "edges"}
                or not _valid_display_text(item.get("updated_at"), limit=64)
            ):
                raise WorkflowRevisionError("workflow_history_invalid")
            versions.append(
                _validated_graph(
                    name=item.get("name"),
                    nodes=item.get("nodes"),
                    edges=item.get("edges"),
                    revision=item.get("revision"),
                )
            )

        revisions = [item["revision"] for item in versions]
        if revisions and (
            revisions[0] != current_revision - len(versions)
            or revisions[-1] != current_revision - 1
            or any(right != left + 1 for left, right in zip(revisions, revisions[1:]))
        ):
            raise WorkflowRevisionError("workflow_history_invalid")

        baseline = next(
            (item for item in versions if item["revision"] == against_revision),
            None,
        )
        if baseline is None:
            oldest_retained = revisions[0] if revisions else current_revision
            if against_revision < oldest_retained:
                raise WorkflowRevisionError("revision_not_retained")
            raise WorkflowRevisionError("workflow_history_invalid")

    before_nodes = {node["id"]: node["type"] for node in baseline["nodes"]}
    current_nodes = {node["id"]: node["type"] for node in current["nodes"]}
    before_edges = {
        (edge["source"], edge["target"]) for edge in baseline["edges"]
    }
    current_edges = {
        (edge["source"], edge["target"]) for edge in current["edges"]
    }

    added_node_ids = sorted(current_nodes.keys() - before_nodes.keys())
    removed_node_ids = sorted(before_nodes.keys() - current_nodes.keys())
    changed_nodes = [
        {
            "node_id": node_id,
            "from_type": before_nodes[node_id],
            "to_type": current_nodes[node_id],
        }
        for node_id in sorted(before_nodes.keys() & current_nodes.keys())
        if before_nodes[node_id] != current_nodes[node_id]
    ]
    added_edges = [
        {"source": source, "target": target}
        for source, target in sorted(current_edges - before_edges)
    ]
    removed_edges = [
        {"source": source, "target": target}
        for source, target in sorted(before_edges - current_edges)
    ]
    name_changed = baseline["name"] != current["name"]
    changed = bool(
        name_changed
        or added_node_ids
        or removed_node_ids
        or changed_nodes
        or added_edges
        or removed_edges
    )

    return {
        "against_revision": against_revision,
        "current_revision": current_revision,
        "status": "changed" if changed else "unchanged",
        "changed": changed,
        "name_changed": name_changed,
        "name_change": (
            {"from": baseline["name"], "to": current["name"]}
            if name_changed
            else None
        ),
        "added_node_ids": added_node_ids,
        "removed_node_ids": removed_node_ids,
        "changed_nodes": changed_nodes,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "baseline_execution_status": "not_executable",
        "current_execution_status": "not_executable",
    }
