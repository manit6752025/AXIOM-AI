from __future__ import annotations

import uuid
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.workflows import (
    WORKFLOW_SCHEMA_VERSION,
    WorkflowStore,
    WorkflowStoreFullError,
    validate_workflow_graph,
)
from axiom.core.workflow_planning import build_workflow_planning_preview
from axiom.version import __version__

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

WORKFLOW_TEMPLATES = [
    {
        "id": "supervised-finetuning",
        "name": "Supervised fine-tuning plan",
        "description": "Review data and model, create a plan, then reserve evaluation and report steps.",
        "nodes": [
            {"id": "dataset", "type": "dataset.profile"},
            {"id": "model", "type": "model.inspect"},
            {"id": "plan", "type": "training.plan"},
            {"id": "evaluation", "type": "evaluation.run"},
            {"id": "report", "type": "report.export"},
        ],
        "edges": [
            {"source": "dataset", "target": "model"},
            {"source": "model", "target": "plan"},
            {"source": "plan", "target": "evaluation"},
            {"source": "evaluation", "target": "report"},
        ],
    },
    {
        "id": "dataset-audit",
        "name": "Dataset quality audit",
        "description": "Profile a local dataset and reserve a report step.",
        "nodes": [
            {"id": "dataset", "type": "dataset.profile"},
            {"id": "report", "type": "report.export"},
        ],
        "edges": [{"source": "dataset", "target": "report"}],
    },
]


class WorkflowNodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: Literal[
        "dataset.profile",
        "model.inspect",
        "training.plan",
        "evaluation.run",
        "report.export",
    ]


class WorkflowEdgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    target: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class WorkflowInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    nodes: list[WorkflowNodeInput] = Field(min_length=1, max_length=64)
    edges: list[WorkflowEdgeInput] = Field(max_length=128)


class WorkflowDryRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter_billions: float = Field(default=7.0, gt=0, le=1000)
    method: Literal["auto", "qlora", "lora", "full"] = "auto"
    seed: int = Field(default=42, ge=0, le=4_294_967_295)


def _validated_payload(request: WorkflowInput) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes = [node.model_dump() for node in request.nodes]
    edges = [edge.model_dump() for edge in request.edges]
    try:
        validate_workflow_graph(request.name, nodes, edges)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return nodes, edges


@router.post("/validate")
def validate_workflow(request: WorkflowInput) -> dict[str, Any]:
    nodes, edges = _validated_payload(request)
    return {"valid": True, "node_count": len(nodes), "connection_count": len(edges)}


@router.get("/templates")
def list_workflow_templates() -> dict[str, Any]:
    return {"templates": WORKFLOW_TEMPLATES}


@router.get("")
def list_workflows(
    limit: int = Query(default=20, ge=1, le=50),
    after_workflow_id: str | None = Query(default=None, min_length=1, max_length=64),
    q: str = Query(default="", max_length=120),
) -> dict[str, Any]:
    """Return cursor-paginated workflow designs without executing them."""
    try:
        return WorkflowStore().list_page(
            limit=limit,
            after_workflow_id=after_workflow_id,
            search=q,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail="Workflow cursor is invalid for the selected search.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow history is invalid.") from exc


@router.post("", status_code=201)
def create_workflow(request: WorkflowInput) -> dict[str, Any]:
    nodes, edges = _validated_payload(request)
    now = datetime.now(UTC).isoformat()
    record = {
        "id": uuid.uuid4().hex,
        "name": request.name,
        "created_at": now,
        "updated_at": now,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "revision": 1,
        "versions": [],
        "nodes": nodes,
        "edges": edges,
        "execution_status": "not_executable",
    }
    try:
        WorkflowStore().save(record)
    except WorkflowStoreFullError as exc:
        raise HTTPException(status_code=409, detail="Workflow history is full.") from exc
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow could not be saved.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("workflows.saved", "Saved a local workflow design")
    return {"workflow": record}


@router.get("/{workflow_id}")
def get_workflow(workflow_id: str) -> dict[str, Any]:
    try:
        workflow = WorkflowStore().get(workflow_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow history is invalid.") from exc
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return {"workflow": workflow}


@router.get("/{workflow_id}/versions")
def list_workflow_versions(workflow_id: str) -> dict[str, Any]:
    try:
        versions = WorkflowStore().versions(workflow_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow history is invalid.") from exc
    if versions is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return {"versions": versions}


@router.put("/{workflow_id}")
def update_workflow(workflow_id: str, request: WorkflowInput) -> dict[str, Any]:
    nodes, edges = _validated_payload(request)
    store = WorkflowStore()
    try:
        record = store.update(
            workflow_id,
            name=request.name,
            nodes=nodes,
            edges=edges,
            updated_at=datetime.now(UTC).isoformat(),
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Workflow not found.")
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow could not be saved.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("workflows.updated", "Updated a local workflow design")
    return {"workflow": record}


@router.post("/{workflow_id}/duplicate", status_code=201)
def duplicate_workflow(workflow_id: str) -> dict[str, Any]:
    store = WorkflowStore()
    try:
        source = store.get(workflow_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Workflow not found.")
        record = store.duplicate(
            workflow_id,
            new_id=uuid.uuid4().hex,
            name=f"{source['name']} (copy)"[:120],
        )
    except WorkflowStoreFullError as exc:
        raise HTTPException(status_code=409, detail="Workflow history is full.") from exc
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow could not be duplicated.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("workflows.duplicated", "Duplicated a local workflow design")
    return {"workflow": record}


@router.post("/{workflow_id}/dry-run")
def dry_run_workflow(workflow_id: str, request: WorkflowDryRunInput) -> dict[str, Any]:
    """Return a resource and input summary without creating or executing a job."""
    try:
        workflow = WorkflowStore().get(workflow_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workflow history is invalid.") from exc
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")

    try:
        preview = build_workflow_planning_preview(
            workflow_id,
            workflow,
            parameter_billions=request.parameter_billions,
            method=request.method,
        )
        hardware = preview.hardware
        plan = preview.training_plan
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Workflow cannot be dry-run until graph validation passes.") from exc

    unresolved_inputs = preview.unresolved_inputs
    summary = {
        "available_hardware": hardware.__dict__,
        "estimated_training_plan": plan.__dict__ if plan is not None else None,
        "estimated_vram_gb": plan.estimated_vram_gb if plan is not None else None,
        "fits_hardware": plan.fits_hardware if plan is not None else None,
        "parameters_billions": request.parameter_billions if plan is not None else None,
        "method": plan.method if plan is not None else None,
    }
    graph_digest = hashlib.sha256(json.dumps(
        {
            "id": workflow_id,
            "revision": preview.workflow_revision,
            "nodes": preview.nodes,
            "edges": preview.edges,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    dependencies: dict[str, str | None] = {}
    for distribution in ("torch", "transformers", "accelerate", "peft", "bitsandbytes"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    reproducibility = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "workflow_id": workflow_id,
        "workflow_revision": workflow["revision"],
        "graph_sha256": graph_digest,
        "seed": request.seed,
        "seed_applied": False,
        "execution_status": "not_executable",
        "software": {
            "axiom": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "dependencies": dependencies,
        },
        "hardware": hardware.__dict__,
        "planning_settings": {
            "parameter_billions": request.parameter_billions if plan is not None else None,
            "requested_method": request.method if plan is not None else None,
            "selected_method": plan.method if plan is not None else None,
        },
    }
    result = {
        "workflow_id": workflow_id,
        "workflow_revision": preview.workflow_revision,
        "schema_version": preview.schema_version,
        "status": "planning_only",
        "execution_status": "not_executable",
        "executed": False,
        "job_created": False,
        "node_count": len(preview.nodes),
        "steps": [
            {"node_id": node["id"], "type": node["type"], "state": "not_run"}
            for node in preview.nodes
        ],
        "unresolved_inputs": unresolved_inputs,
        "resource_summary": summary,
        "reproducibility_manifest": reproducibility,
        "warnings": [
            "This is a planning-only dry run; no training, evaluation, report, or other job was created or executed.",
            "Resource values are estimates based on current detected hardware and the requested training-plan settings.",
        ],
    }
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("workflows.dry_run", f"Planned workflow {workflow_id} without execution")
    return {"dry_run": result}
