from __future__ import annotations

import os
import re
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from axiom.core.hardware import detect_hardware
from axiom.core.experiments import ExperimentStore
from axiom.datasets.profiler import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_ROWS,
    profile_jsonl_stream,
)
from axiom.datasets.manifest import DEFAULT_MANIFEST_MAX_BYTES, create_dataset_manifest
from axiom.datasets.cleaning_preview import preview_jsonl_cleaning
from axiom.datasets.quality import compare_dataset_schemas, validate_dataset_profile
from axiom.models.registry import Model, ModelRegistry
from axiom.models.compatibility import estimate_model_compatibility
from axiom.models.discovery import discover_local_model_folders
from axiom.models.inspector import inspect_model
from axiom.api.model_paths import validate_model_path
from axiom.path_safety import (
    DEFAULT_SKIP_DIRECTORIES,
    SafePathError,
    find_existing_relative_path,
    open_regular_file,
)
from axiom.training.planner import create_training_plan

router = APIRouter(prefix="/api/workspace", tags=["workspace"])
_events: deque[dict[str, Any]] = deque(maxlen=100)
_events_lock = Lock()
_allowed_actions = {
    "dashboard.module_details",
    "models.add",
    "models.refresh",
    "models.update",
    "models.remove",
    "models.import",
    "datasets.import",
    "datasets.refresh",
    "datasets.profile",
    "datasets.compare",
    "experiments.update",
    "experiments.remove",
    "training.new_plan",
    "evaluation.run",
    "evaluation.remove",
    "runtime.connect",
    "runtime.explain_empty",
    "mcp.add_server",
    "mcp.inspect",
    "mcp.check_contract",
    "logs.filter",
    "logs.explain_empty",
    "settings.integration_note",
}


class WorkspaceAction(BaseModel):
    action: str = Field(min_length=1, max_length=64)
    detail: str | None = Field(default=None, max_length=500)


class WorkspaceModelInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1, max_length=512)
    format: str = Field(default="unknown", min_length=1, max_length=64)
    parameters: str | None = Field(default=None, max_length=80)
    quantization: str | None = Field(default=None, max_length=80)
    architecture: str | None = Field(default=None, max_length=160)
    license: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=32)
    favorite: bool = False
    content_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class WorkspaceModelPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str | None = Field(default=None, max_length=64)
    parameters: str | None = Field(default=None, max_length=80)
    quantization: str | None = Field(default=None, max_length=80)
    architecture: str | None = Field(default=None, max_length=160)
    license: str | None = Field(default=None, max_length=120)
    tags: list[str] | None = Field(default=None, max_length=32)
    favorite: bool | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")


class WorkspaceModelImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    models: list[WorkspaceModelInput] = Field(max_length=1000)


class WorkspaceModelInspectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=512)


class WorkspaceTrainingPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    precision: str
    lora_rank: int
    batch_size: int
    gradient_accumulation: int
    sequence_length: int
    learning_rate: float
    estimated_vram_gb: float | None
    fits_hardware: bool | None
    reason: str


class WorkspaceExperimentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    parameter_billions: float = Field(gt=0, le=1000)
    method: Literal["auto", "qlora", "lora", "full"] = "auto"
    plan_snapshot: WorkspaceTrainingPlanInput | None = None
    model: str | None = Field(default=None, max_length=256)
    dataset: str | None = Field(default=None, max_length=512)
    notes: str | None = Field(default=None, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in tags]
        if any(not tag or len(tag) > 64 for tag in normalized):
            raise ValueError("Experiment tags must contain 1 to 64 characters.")
        if len({tag.casefold() for tag in normalized}) != len(normalized):
            raise ValueError("Experiment tags must be unique.")
        return normalized


class WorkspaceExperimentMetadataInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in tags]
        if any(not tag or len(tag) > 64 for tag in normalized):
            raise ValueError("Experiment tags must contain 1 to 64 characters.")
        if len({tag.casefold() for tag in normalized}) != len(normalized):
            raise ValueError("Experiment tags must be unique.")
        return normalized


class WorkspaceDatasetQualityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(min_length=2, max_length=3)
    required_fields: list[str] = Field(default_factory=list, max_length=100)
    expected_types: dict[str, Literal[
        "string", "integer", "number", "boolean", "object", "array", "null",
    ]] = Field(default_factory=dict, max_length=100)
    duplicate_key_fields: list[str] = Field(default_factory=list, max_length=100)
    duplicate_rate_threshold: float | None = Field(default=None, ge=0, le=1)
    text_fields: list[str] = Field(default_factory=list, max_length=100)
    sensitive_field_rules: list[str] = Field(default_factory=list, max_length=100)
    ignored_sensitive_fields: list[str] = Field(default_factory=list, max_length=100)
    scan_pii: bool = False
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, ge=1, le=256 * 1024 * 1024)
    max_rows: int = Field(default=DEFAULT_MAX_ROWS, ge=1, le=500_000)
    max_line_bytes: int = Field(default=DEFAULT_MAX_LINE_BYTES, ge=1, le=8 * 1024 * 1024)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        if any(not path or len(path) > 512 for path in paths):
            raise ValueError("Dataset paths must be between 1 and 512 characters.")
        return paths

    @field_validator("required_fields")
    @classmethod
    def validate_required_fields(cls, fields: list[str]) -> list[str]:
        normalized = [field.strip() for field in fields]
        if any(not field or len(field) > 160 for field in normalized):
            raise ValueError("Required field names must be between 1 and 160 characters.")
        return normalized

    @field_validator("expected_types")
    @classmethod
    def validate_expected_types(cls, fields: dict[str, str]) -> dict[str, str]:
        normalized = {name.strip(): value for name, value in fields.items()}
        if any(not name or len(name) > 160 for name in normalized):
            raise ValueError("Expected-type field names must be between 1 and 160 characters.")
        if len(normalized) != len(fields):
            raise ValueError("Expected-type field names must be unique after trimming.")
        return normalized

    @field_validator(
        "duplicate_key_fields", "text_fields", "sensitive_field_rules", "ignored_sensitive_fields",
    )
    @classmethod
    def validate_profile_fields(cls, fields: list[str]) -> list[str]:
        normalized = [name.strip() for name in fields]
        if any(not name or len(name) > 160 for name in normalized):
            raise ValueError("Profile field names must be between 1 and 160 characters.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Profile field names must be unique after trimming.")
        return normalized


class WorkspaceDatasetCleaningPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=512)
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, ge=1, le=256 * 1024 * 1024)
    max_rows: int = Field(default=DEFAULT_MAX_ROWS, ge=1, le=500_000)
    max_line_bytes: int = Field(default=DEFAULT_MAX_LINE_BYTES, ge=1, le=8 * 1024 * 1024)


def record_workspace_event(action: str, detail: str | None = None) -> dict[str, Any]:
    event = {
        "action": action,
        "detail": detail,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with _events_lock:
        _events.appendleft(event)
    return event


@router.post("/actions")
def record_action(request: WorkspaceAction) -> dict[str, Any]:
    if request.action not in _allowed_actions:
        raise HTTPException(status_code=400, detail="Unsupported workspace action")
    event = record_workspace_event(request.action, request.detail)
    return {"status": "accepted", "event": event}


@router.get("/events")
def workspace_events(
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100),
    search: str = Query(default="", max_length=160),
) -> dict[str, Any]:
    """Return a bounded, newest-first page of recent workspace events."""
    with _events_lock:
        events = list(_events)
    needle = search.strip().casefold()
    if needle:
        events = [
            event for event in events
            if needle in str(event.get("action", "")).casefold()
            or needle in str(event.get("detail") or "").casefold()
        ]
    total = len(events)
    return {
        "events": events[offset:offset + limit],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
        "filters": {"search": needle},
    }


def _model_parameter_sort_key(value: str | None) -> tuple[bool, float]:
    if not value:
        return True, 0.0
    match = re.search(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*([KMBT]?)", value, re.IGNORECASE)
    if not match:
        return True, 0.0
    scale = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0, "T": 1_000_000_000_000.0}
    return False, float(match.group(1)) * scale[match.group(2).upper()]


@router.get("/models")
def workspace_models(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
    search: str = Query(default="", max_length=160),
    favorite_only: bool = False,
    format: str | None = Query(default=None, max_length=64),
    tag: str | None = Query(default=None, max_length=40),
    sort_by: Literal["name", "source", "format", "parameters", "quantization", "favorite"] = "name",
    sort_order: Literal["asc", "desc"] = "asc",
) -> dict[str, Any]:
    """Return a bounded, searchable, filterable and stable-sorted model registry page."""
    models = ModelRegistry().list()
    needle = search.strip().casefold()
    if needle:
        def matches(model: Model) -> bool:
            fields = (
                model.name, model.source, model.format, model.parameters,
                model.quantization, model.architecture, model.license, *model.tags,
            )
            return any(needle in value.casefold() for value in fields if isinstance(value, str))
        models = [model for model in models if matches(model)]
    if favorite_only:
        models = [model for model in models if model.favorite]
    if format and format.strip() and format.strip().casefold() != "all":
        wanted_format = format.strip().casefold()
        models = [model for model in models if model.format.casefold() == wanted_format]
    if tag and tag.strip():
        wanted_tag = tag.strip().casefold()
        models = [model for model in models if any(item.casefold() == wanted_tag for item in model.tags)]

    if sort_by == "parameters":
        sized_models = [model for model in models if not _model_parameter_sort_key(model.parameters)[0]]
        unsized_models = [model for model in models if _model_parameter_sort_key(model.parameters)[0]]
        # Sort by name first so equal parameter counts keep an intuitive
        # alphabetical order in both ascending and descending size modes.
        sized_models.sort(key=lambda model: model.name.casefold())
        sized_models.sort(
            key=lambda model: _model_parameter_sort_key(model.parameters)[1],
            reverse=sort_order == "desc",
        )
        unsized_models.sort(key=lambda model: model.name.casefold())
        # Unknown parameter metadata stays last for both sort directions.
        models = [*sized_models, *unsized_models]
    else:
        def sort_key(model: Model) -> tuple[Any, ...]:
            if sort_by == "favorite":
                return model.favorite, model.name.casefold()
            value = getattr(model, sort_by)
            normalized = value.casefold() if isinstance(value, str) else ""
            return normalized, model.name.casefold()

        models.sort(key=sort_key, reverse=sort_order == "desc")
    total = len(models)
    return {
        "models": [model.to_dict() for model in models[offset:offset + limit]],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
        "filters": {
            "search": needle,
            "favorite_only": favorite_only,
            "format": format.strip() if format else None,
            "tag": tag.strip() if tag else None,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
    }


@router.post("/models", status_code=201)
def workspace_add_model(request: WorkspaceModelInput) -> dict[str, Any]:
    try:
        model = Model(**request.model_dump(exclude_none=True))
        ModelRegistry().add(model)
    except (TypeError, ValueError) as exc:
        if "already exists" in str(exc) or "already registered" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record_workspace_event("models.add", f"Registered {model.name}")
    return {"model": model.to_dict()}


@router.get("/models/export")
def workspace_export_models() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "models": [model.to_dict() for model in ModelRegistry().list()],
    }


@router.get("/models/discover")
def workspace_discover_models() -> dict[str, Any]:
    try:
        return discover_local_model_folders(Path.cwd() / "models")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Local model discovery failed safely.") from exc


@router.post("/models/inspect")
def workspace_inspect_model(request: WorkspaceModelInspectInput) -> dict[str, Any]:
    try:
        model_path = validate_model_path(str(Path.cwd() / "models" / request.path))
        inspection = inspect_model(str(model_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Local model folder not found.") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Model inspection rejected the selected path.") from exc
    return {
        "inspection": {
            "path": model_path.relative_to(Path.cwd().resolve()).as_posix(),
            "architecture": inspection.architecture,
            "model_type": inspection.model_type,
            "parameter_count": inspection.parameter_count,
            "parameter_source": inspection.parameter_source,
            "model_format": inspection.model_format,
            "weight_size_bytes": inspection.weight_size_bytes,
            "has_config": inspection.has_config,
            "has_tokenizer": inspection.has_tokenizer,
            "has_safetensors": inspection.has_safetensors,
            "has_gguf": inspection.has_gguf,
            "has_pytorch_weights": inspection.has_pytorch_weights,
            "capabilities": inspection.capabilities,
            "estimated_vram_gb": inspection.estimated_vram_gb,
            "status": inspection.status,
        },
    }


@router.post("/models/import", status_code=201)
def workspace_import_models(request: WorkspaceModelImportInput) -> dict[str, Any]:
    try:
        imported = ModelRegistry().import_models([
            model.model_dump(exclude_none=True) for model in request.models
        ])
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Model metadata could not be imported.") from exc
    except (TypeError, ValueError) as exc:
        status = 409 if "already" in str(exc) or "registered" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    record_workspace_event("models.import", f"Imported {len(imported)} model records")
    return {"models": [model.to_dict() for model in imported], "imported": len(imported)}


@router.get("/models/{model_id}")
def workspace_model(model_id: str) -> dict[str, Any]:
    model = ModelRegistry().get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return {"model": model.to_dict()}


@router.get("/models/{model_id}/compatibility")
def workspace_model_compatibility(model_id: str) -> dict[str, Any]:
    """Estimate memory fit and report metadata-backed compatibility guidance."""
    try:
        model = ModelRegistry().get(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found.")
        report = estimate_model_compatibility(model, detect_hardware())
    except HTTPException:
        raise
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Model compatibility report could not be generated.") from exc
    record_workspace_event("models.compatibility", f"Checked fit metadata for {model.name}")
    return {"report": report}


@router.patch("/models/{model_id}")
def workspace_update_model(model_id: str, request: WorkspaceModelPatch) -> dict[str, Any]:
    changes = request.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="At least one metadata field is required.")
    if any(value is None for key, value in changes.items() if key not in {"license", "content_sha256", "architecture", "parameters", "quantization", "tags"}):
        raise HTTPException(status_code=422, detail="Non-null metadata values are required.")
    try:
        model = ModelRegistry().update(model_id, changes)
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Model metadata could not be updated.") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    record_workspace_event("models.update", f"Updated metadata for {model.name}")
    return {"model": model.to_dict()}


@router.delete("/models/{model_id}")
def workspace_remove_model(model_id: str) -> dict[str, Any]:
    try:
        model = ModelRegistry().remove(model_id)
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Model registry could not be updated.") from exc
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    record_workspace_event("models.remove", f"Removed registry entry for {model.name}")
    return {"removed": model.to_dict(), "weights_deleted": False}


@router.get("/experiments")
def workspace_experiments(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
    q: str | None = Query(default=None, max_length=120),
    tag: str | None = Query(default=None, min_length=1, max_length=64),
) -> dict[str, Any]:
    try:
        page = ExperimentStore().list_page(limit=limit, offset=offset, query=q, tag=tag)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Experiment history is invalid.") from exc
    return page


@router.get("/experiments/{experiment_id}")
def workspace_experiment(experiment_id: str) -> dict[str, Any]:
    try:
        record = ExperimentStore().get(experiment_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Experiment history is invalid.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"experiment": record}


@router.patch("/experiments/{experiment_id}")
def workspace_update_experiment(
    experiment_id: str,
    request: WorkspaceExperimentMetadataInput,
) -> dict[str, Any]:
    try:
        record = ExperimentStore().update_metadata(
            experiment_id,
            name=request.name,
            notes=request.notes,
            tags=request.tags,
        )
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    record_workspace_event("experiments.update", f"Updated metadata for plan {experiment_id}")
    return {"experiment": record, "plan_immutable": True}


@router.delete("/experiments/{experiment_id}")
def workspace_delete_experiment(experiment_id: str) -> dict[str, Any]:
    try:
        record = ExperimentStore().delete(experiment_id)
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Experiment history could not be updated.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    record_workspace_event("experiments.remove", f"Deleted plan record {experiment_id}")
    return {"deleted": record, "run_artifacts_deleted": False}


@router.post("/experiments", status_code=201)
def workspace_add_experiment(request: WorkspaceExperimentInput) -> dict[str, Any]:
    try:
        hardware = detect_hardware()
        plan = create_training_plan(
            request.parameter_billions,
            hardware,
            request.method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.plan_snapshot is not None and request.plan_snapshot.model_dump() != plan.__dict__:
        raise HTTPException(
            status_code=409,
            detail="The displayed plan no longer matches current hardware. Rebuild the plan before saving it.",
        )

    record = {
        "schema_version": 1,
        "id": uuid.uuid4().hex,
        "name": request.name,
        "created_at": datetime.now(UTC).isoformat(),
        "model": request.model,
        "dataset": request.dataset,
        "notes": request.notes,
        "tags": request.tags,
        "parameter_billions": request.parameter_billions,
        "requested_method": request.method,
        "status": "planned",
        "hardware": hardware.__dict__,
        "plan": plan.__dict__,
    }
    try:
        ExperimentStore().add(record)
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Experiment could not be saved.") from exc
    record_workspace_event("experiments.saved", f"Saved plan: {request.name}")
    return {"experiment": record}


@router.get("/datasets")
def workspace_datasets(
    path: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=500, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=5000),
    search: str = Query(default="", max_length=160),
) -> dict[str, Any]:
    """List a bounded, stable page of JSONL dataset metadata.

    Discovery is capped at 5,000 entries. The response reports truncation
    separately so clients never mistake a bounded inventory for a full scan.
    """
    def page_result(items: list[dict[str, Any]], *, truncated: bool = False) -> dict[str, Any]:
        needle = search.strip().casefold()
        filtered = [
            item for item in items
            if not needle or needle in str(item.get("path", "")).casefold()
        ]
        filtered.sort(key=lambda item: str(item.get("path", "")).casefold())
        total = len(filtered)
        return {
            "datasets": filtered[offset:offset + limit],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
            "truncated": truncated,
            "scan_limit": 5000,
            "filters": {"path": path, "search": needle},
        }

    workspace_root = Path.cwd().resolve()
    data_entry = workspace_root / "data"
    if data_entry.is_symlink():
        raise HTTPException(status_code=400, detail="The data directory must not be a symlink.")
    try:
        root = data_entry.resolve(strict=True)
    except FileNotFoundError:
        return page_result([])
    if not root.is_dir() or not root.is_relative_to(workspace_root):
        raise HTTPException(status_code=400, detail="Invalid data directory.")

    candidate = root
    if path:
        relative = path
        while relative.startswith("./"):
            relative = relative[2:]
        if relative == "data":
            relative = ""
        elif relative.startswith("data/"):
            relative = relative[5:]

        if relative:
            try:
                candidate = find_existing_relative_path(
                    root, relative, skip_directories=DEFAULT_SKIP_DIRECTORIES
                )
            except SafePathError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileNotFoundError:
                return page_result([])

    datasets: list[dict[str, Any]] = []
    truncated = False
    if candidate.is_file():
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise HTTPException(status_code=400, detail="Invalid dataset path.")
            datasets.append({
                "path": resolved.relative_to(root).as_posix(),
                "size_bytes": resolved.stat().st_size,
            })
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Dataset path could not be inspected safely.") from exc
    elif candidate.is_dir():
        for current, directories, filenames in os.walk(candidate, followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(
                name for name in directories
                if name not in DEFAULT_SKIP_DIRECTORIES
                and not (current_path / name).is_symlink()
            )
            for name in sorted(filenames):
                if not name.lower().endswith(".jsonl"):
                    continue
                item = current_path / name
                if item.is_symlink():
                    continue
                try:
                    resolved = item.resolve(strict=True)
                    if not resolved.is_relative_to(root) or not resolved.is_file():
                        continue
                    size = resolved.stat().st_size
                    relative_name = resolved.relative_to(root).as_posix()
                except (OSError, RuntimeError, ValueError):
                    continue
                datasets.append({"path": relative_name, "size_bytes": size})
                if len(datasets) > 5000:
                    truncated = True
                    break
            if truncated:
                break

    return page_result(datasets[:5000], truncated=truncated)
@router.post("/training-plan")
def workspace_training_plan(
    parameter_billions: float = Query(default=7.0, gt=0, le=1000),
    method: Literal["auto", "qlora", "lora", "full"] = Query(default="auto"),
) -> dict[str, Any]:
    try:
        plan = create_training_plan(parameter_billions, detect_hardware(), method)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record_workspace_event(
        "training.new_plan",
        f"{parameter_billions:g}B model · {plan.method} · {plan.precision}",
    )
    return {"plan": plan.__dict__}


@router.get("/runtime")
def workspace_runtime() -> dict[str, Any]:
    hardware = detect_hardware()
    return {"status": "offline", "hardware": hardware.__dict__}

@router.get("/dataset-profile")
def workspace_dataset_profile(
    path: str = Query(..., min_length=1, max_length=512),
    max_bytes: int = Query(default=DEFAULT_MAX_BYTES, ge=1, le=256 * 1024 * 1024),
    max_rows: int = Query(default=DEFAULT_MAX_ROWS, ge=1, le=500_000),
    max_line_bytes: int = Query(default=DEFAULT_MAX_LINE_BYTES, ge=1, le=8 * 1024 * 1024),
    duplicate_key_fields: list[str] = Query(default=[]),
    duplicate_rate_threshold: float | None = Query(default=None, ge=0, le=1),
    text_fields: list[str] = Query(default=[]),
    sensitive_field_rules: list[str] = Query(default=[]),
    ignored_sensitive_fields: list[str] = Query(default=[]),
    scan_pii: bool = Query(default=False),
) -> dict[str, Any]:
    """Profile a JSONL file selected from the local data directory."""
    dataset = _profile_workspace_dataset(
        path,
        max_bytes=max_bytes,
        max_rows=max_rows,
        max_line_bytes=max_line_bytes,
        duplicate_key_fields=duplicate_key_fields,
        duplicate_rate_threshold=duplicate_rate_threshold,
        text_fields=text_fields,
        sensitive_field_rules=sensitive_field_rules,
        ignored_sensitive_fields=ignored_sensitive_fields,
        scan_pii=scan_pii,
    )
    record_workspace_event("datasets.profile", f"Profiled {dataset['path']}")
    return {"profile": dataset["profile"]}


@router.get("/dataset-manifest")
def workspace_dataset_manifest(
    path: str = Query(..., min_length=1, max_length=512),
    max_bytes: int = Query(default=DEFAULT_MANIFEST_MAX_BYTES, ge=1, le=1024 * 1024 * 1024),
    parent_manifest_sha256: str | None = Query(default=None, pattern=r"^[0-9a-fA-F]{64}$"),
) -> dict[str, Any]:
    """Create a value-free checksum/provenance manifest for a local JSONL file."""
    root = _workspace_dataset_root()
    candidate = _resolve_workspace_dataset(root, path)
    try:
        with open_regular_file(candidate) as stream:
            size = os.fstat(stream.fileno()).st_size
            manifest = create_dataset_manifest(
                stream,
                file_name=candidate.name,
                relative_path=candidate.relative_to(root).as_posix(),
                file_size_bytes=size,
                max_bytes=max_bytes,
                parent_manifest_sha256=parent_manifest_sha256,
            )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Dataset manifest could not be created.") from exc
    record_workspace_event("datasets.manifest", f"Hashed {manifest['bytes_hashed']} bytes for {manifest['path']}")
    return {"manifest": manifest}


@router.post("/dataset-cleaning-preview")
def workspace_dataset_cleaning_preview(
    request: WorkspaceDatasetCleaningPreviewInput,
) -> dict[str, Any]:
    """Preview the existing cleaner's effects without creating or changing files."""
    root = _workspace_dataset_root()
    candidate = _resolve_workspace_dataset(root, request.path)
    try:
        with open_regular_file(candidate) as stream:
            size = os.fstat(stream.fileno()).st_size
            preview = preview_jsonl_cleaning(
                stream,
                file_name=candidate.name,
                file_size_bytes=size,
                max_bytes=request.max_bytes,
                max_rows=request.max_rows,
                max_line_bytes=request.max_line_bytes,
            )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Dataset cleaning preview could not be created.") from exc
    preview["path"] = candidate.relative_to(root).as_posix()
    record_workspace_event("datasets.clean_preview", f"Previewed {preview['path']} without writing output")
    return {"preview": preview}


@router.post("/dataset-quality")
def workspace_dataset_quality(request: WorkspaceDatasetQualityInput) -> dict[str, Any]:
    """Compare two or three local dataset schemas and run structural checks."""
    if len(set(request.paths)) != len(request.paths):
        raise HTTPException(status_code=422, detail="Each dataset must be selected only once.")
    if request.max_bytes * len(request.paths) > 512 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Combined dataset scan budget cannot exceed 512 MB.")

    datasets = [
        _profile_workspace_dataset(
            path,
            max_bytes=request.max_bytes,
            max_rows=request.max_rows,
            max_line_bytes=request.max_line_bytes,
            duplicate_key_fields=request.duplicate_key_fields,
            duplicate_rate_threshold=request.duplicate_rate_threshold,
            text_fields=request.text_fields,
            sensitive_field_rules=request.sensitive_field_rules,
            ignored_sensitive_fields=request.ignored_sensitive_fields,
            scan_pii=request.scan_pii,
        )
        for path in request.paths
    ]
    try:
        comparison = compare_dataset_schemas(datasets)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    checks = {
        item["path"]: validate_dataset_profile(
            item["profile"],
            required_fields=request.required_fields,
            expected_types=request.expected_types,
        )
        for item in datasets
    }
    record_workspace_event(
        "datasets.compare",
        f"Compared {len(datasets)} dataset schemas",
    )
    return {
        "datasets": datasets,
        "comparison": comparison,
        "checks": checks,
        "limits": {
            "max_bytes": request.max_bytes,
            "max_rows": request.max_rows,
            "max_line_bytes": request.max_line_bytes,
        },
    }


def _workspace_dataset_root() -> Path:
    workspace_root = Path.cwd().resolve()
    data_entry = workspace_root / "data"
    if data_entry.is_symlink():
        raise HTTPException(status_code=400, detail="The data directory must not be a symlink.")
    try:
        root = data_entry.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset file not found.") from exc
    if not root.is_dir() or not root.is_relative_to(workspace_root):
        raise HTTPException(status_code=400, detail="Invalid data directory.")
    return root


def _resolve_workspace_dataset(root: Path, path: str) -> Path:
    relative = path
    while relative.startswith("./"):
        relative = relative[2:]
    if relative.startswith("data/"):
        relative = relative[5:]

    if not relative or not relative.rpartition("/")[2].lower().endswith(".jsonl"):
        raise HTTPException(status_code=422, detail="Only .jsonl files are supported.")

    try:
        candidate = find_existing_relative_path(
            root,
            relative,
            allow_directories=False,
            skip_directories=DEFAULT_SKIP_DIRECTORIES,
        )
    except SafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset file not found.") from exc
    return candidate


def _profile_workspace_dataset(
    path: str,
    *,
    max_bytes: int,
    max_rows: int,
    max_line_bytes: int,
    duplicate_key_fields: list[str] | tuple[str, ...] = (),
    duplicate_rate_threshold: float | None = None,
    text_fields: list[str] | tuple[str, ...] = (),
    sensitive_field_rules: list[str] | tuple[str, ...] = (),
    ignored_sensitive_fields: list[str] | tuple[str, ...] = (),
    scan_pii: bool = False,
) -> dict[str, Any]:
    root = _workspace_dataset_root()
    candidate = _resolve_workspace_dataset(root, path)
    try:
        with open_regular_file(candidate) as stream:
            size = os.fstat(stream.fileno()).st_size
            report = profile_jsonl_stream(
                stream,
                file_name=candidate.name,
                file_size_bytes=size,
                max_bytes=max_bytes,
                max_rows=max_rows,
                max_line_bytes=max_line_bytes,
                duplicate_key_fields=duplicate_key_fields,
                duplicate_rate_threshold=duplicate_rate_threshold,
                text_fields=text_fields,
                sensitive_field_rules=sensitive_field_rules,
                ignored_sensitive_fields=ignored_sensitive_fields,
                scan_pii=scan_pii,
            )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Dataset could not be profiled.") from exc
    return {"path": candidate.relative_to(root).as_posix(), "profile": report}
