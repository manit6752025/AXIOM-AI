"""API for bounded offline evaluation of already-produced JSONL outputs."""

from __future__ import annotations

import os
import platform
import stat
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from axiom.api.workspace import _resolve_workspace_dataset, _workspace_dataset_root
from axiom.evaluation.jsonl import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_ROWS,
    evaluate_jsonl_stream,
)
from axiom.evaluation.service import (
    EvaluationBaselineNotFoundError,
    EvaluationBaselinePermissionError,
    EvaluationHistoryReadError,
    EvaluationPersistenceError,
    persist_evaluation_run,
)
from axiom.evaluation.store import EvaluationStore
from axiom.path_safety import open_regular_file

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvaluationRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(min_length=1, max_length=512)
    prediction_field: str = Field(min_length=1, max_length=160)
    reference_field: str = Field(min_length=1, max_length=160)
    split_field: str | None = Field(default=None, max_length=160)
    split_value: str | None = Field(default=None, max_length=160)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    model_label: str | None = Field(default=None, max_length=256)
    baseline: bool = False
    baseline_run_id: str | None = Field(default=None, min_length=1, max_length=64)
    regression_metric: Literal["exact_match", "normalized_exact_match", "token_f1_micro"] = "token_f1_micro"
    regression_tolerance: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, ge=1, le=256 * 1024 * 1024)
    max_rows: int = Field(default=DEFAULT_MAX_ROWS, ge=1, le=500_000)
    max_line_bytes: int = Field(default=DEFAULT_MAX_LINE_BYTES, ge=1, le=8 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_split(self) -> "EvaluationRunInput":
        if (self.split_field is None) != (self.split_value is None):
            raise ValueError("Split field and selected split value must be supplied together.")
        if self.prediction_field == self.reference_field:
            raise ValueError("Prediction and reference fields must be different.")
        return self


@router.post("/runs", status_code=201)
def create_evaluation_run(request: EvaluationRunInput) -> dict[str, Any]:
    root = _workspace_dataset_root()
    candidate = _resolve_workspace_dataset(root, request.path)
    try:
        with open_regular_file(candidate) as stream:
            initial = os.fstat(stream.fileno())
            report = evaluate_jsonl_stream(
                stream,
                path=candidate.relative_to(root).as_posix(),
                file_name=candidate.name,
                file_size_bytes=initial.st_size,
                prediction_field=request.prediction_field,
                reference_field=request.reference_field,
                split_field=request.split_field,
                split_value=request.split_value,
                max_bytes=request.max_bytes,
                max_rows=request.max_rows,
                max_line_bytes=request.max_line_bytes,
            )
            final = os.fstat(stream.fileno())
            current_path = candidate.lstat()
            if (initial.st_size, initial.st_mtime_ns, initial.st_ino) != (
                final.st_size, final.st_mtime_ns, final.st_ino
            ) or stat.S_ISLNK(current_path.st_mode) or (
                initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns
            ) != (
                current_path.st_dev, current_path.st_ino, current_path.st_size, current_path.st_mtime_ns
            ):
                raise HTTPException(status_code=409, detail="Dataset changed during evaluation; retry against a stable file.")
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        record = persist_evaluation_run(
            report,
            name=request.name,
            model_label=request.model_label,
            baseline=request.baseline,
            baseline_run_id=request.baseline_run_id,
            regression_metric=request.regression_metric,
            regression_tolerance=request.regression_tolerance,
            python_version=platform.python_version(),
        )
    except EvaluationBaselineNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (EvaluationBaselinePermissionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (EvaluationHistoryReadError, EvaluationPersistenceError):
        raise HTTPException(status_code=500, detail="Evaluation report could not be saved safely.") from None

    from axiom.api.workspace import record_workspace_event

    record_workspace_event("evaluation.run", f"Evaluated precomputed outputs from {report['path']}")
    return {"run": record}


@router.get("/runs")
def list_evaluation_runs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
) -> dict[str, Any]:
    try:
        return EvaluationStore().list_page(limit=limit, offset=offset)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Evaluation history is invalid.") from exc


@router.get("/runs/{run_id}")
def get_evaluation_run(run_id: str) -> dict[str, Any]:
    try:
        run = EvaluationStore().get(run_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Evaluation history is invalid.") from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return {"run": run}


@router.delete("/runs/{run_id}")
def delete_evaluation_run(run_id: str) -> dict[str, Any]:
    try:
        run = EvaluationStore().delete(run_id)
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Evaluation history could not be updated.") from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("evaluation.remove", f"Deleted evaluation run {run_id}")
    return {"deleted": run}
