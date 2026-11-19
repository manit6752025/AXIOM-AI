"""Read-only, bounded planning previews for saved workflow designs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from axiom.core.hardware import HardwareInfo, detect_hardware
from axiom.core.workflows import WORKFLOW_SCHEMA_VERSION, validate_workflow_graph
from axiom.training.planner import TrainingPlan, create_training_plan

_WORKFLOW_ID = re.compile(r"^[0-9a-f]{32}$")
_TRAINING_METHODS = frozenset({"auto", "qlora", "lora", "full"})
_UNRESOLVED_NODE_TYPES = {
    "dataset.profile",
    "model.inspect",
    "evaluation.run",
    "report.export",
}


@dataclass(frozen=True)
class WorkflowPlanningPreview:
    workflow_id: str
    workflow_revision: int
    schema_version: int
    nodes: list[dict[str, str]]
    edges: list[dict[str, str]]
    hardware: HardwareInfo
    training_plan: TrainingPlan | None
    unresolved_inputs: list[dict[str, str]]


def build_workflow_planning_preview(
    workflow_id: str,
    workflow: dict[str, Any],
    *,
    parameter_billions: float = 7.0,
    method: str = "auto",
    hardware: HardwareInfo | None = None,
) -> WorkflowPlanningPreview:
    """Validate a saved design and estimate its training-plan node, if present.

    This function reads no workspace state and performs no writes or execution.
    Callers supply a record already read from the workflow store.
    """
    if not isinstance(workflow_id, str) or not _WORKFLOW_ID.fullmatch(workflow_id):
        raise ValueError("Workflow ID is invalid.")
    if not isinstance(workflow, dict) or workflow.get("id") != workflow_id:
        raise ValueError("Saved workflow record is malformed.")

    revision = workflow.get("revision")
    schema_version = workflow.get("schema_version")
    name = workflow.get("name")
    raw_nodes = workflow.get("nodes")
    raw_edges = workflow.get("edges")
    if (
        type(revision) is not int
        or not 1 <= revision <= 1_000_000
        or type(schema_version) is not int
        or schema_version != WORKFLOW_SCHEMA_VERSION
        or not isinstance(name, str)
        or type(raw_nodes) is not list
        or not 1 <= len(raw_nodes) <= 64
        or type(raw_edges) is not list
        or len(raw_edges) > 128
    ):
        raise ValueError("Saved workflow record is malformed.")

    nodes: list[dict[str, str]] = []
    for node in raw_nodes:
        if (
            not isinstance(node, dict)
            or set(node) != {"id", "type"}
            or not isinstance(node.get("id"), str)
            or not isinstance(node.get("type"), str)
        ):
            raise ValueError("Saved workflow node is malformed.")
        nodes.append({"id": node["id"], "type": node["type"]})

    edges: list[dict[str, str]] = []
    for edge in raw_edges:
        if (
            not isinstance(edge, dict)
            or set(edge) != {"source", "target"}
            or not isinstance(edge.get("source"), str)
            or not isinstance(edge.get("target"), str)
        ):
            raise ValueError("Saved workflow connection is malformed.")
        edges.append({"source": edge["source"], "target": edge["target"]})

    try:
        validate_workflow_graph(name, nodes, edges)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Saved workflow graph is invalid.") from exc

    if (
        isinstance(parameter_billions, bool)
        or not isinstance(parameter_billions, (int, float))
    ):
        raise ValueError("Parameter count must be a finite number from greater than zero to 1,000.")
    try:
        parameter_value = float(parameter_billions)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Parameter count must be a finite number from greater than zero to 1,000.") from exc
    if not math.isfinite(parameter_value) or not 0 < parameter_value <= 1000:
        raise ValueError("Parameter count must be a finite number from greater than zero to 1,000.")
    if not isinstance(method, str) or method.strip().lower() not in _TRAINING_METHODS:
        raise ValueError("Training method must be auto, qlora, lora, or full.")

    detected_hardware = hardware if hardware is not None else detect_hardware()
    has_training_plan = any(node["type"] == "training.plan" for node in nodes)
    training_plan = (
        create_training_plan(parameter_value, detected_hardware, method.strip().lower())
        if has_training_plan
        else None
    )
    unresolved_inputs = [
        {
            "node_id": node["id"],
            "type": node["type"],
            "reason": "This design node has no saved path or configuration.",
        }
        for node in nodes
        if node["type"] in _UNRESOLVED_NODE_TYPES
    ]

    return WorkflowPlanningPreview(
        workflow_id=workflow_id,
        workflow_revision=revision,
        schema_version=schema_version,
        nodes=nodes,
        edges=edges,
        hardware=detected_hardware,
        training_plan=training_plan,
        unresolved_inputs=unresolved_inputs,
    )
