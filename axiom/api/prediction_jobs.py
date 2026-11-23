"""Workspace-scoped API for bounded, cancellable local prediction jobs."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from axiom.core.prediction_jobs import (
    cancel_prediction_job,
    get_prediction_job,
    list_prediction_jobs,
    resume_prediction_job,
    start_prediction_job,
)
from axiom.core.workspace_context import active_workspace_root

router = APIRouter(prefix="/api/prediction-jobs", tags=["prediction-jobs"])


class PredictionJobStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_path: str = Field(min_length=1, max_length=4096)
    adapter_path: str | None = Field(default=None, min_length=1, max_length=4096)
    dataset_path: str = Field(min_length=1, max_length=512)
    output_path: str = Field(min_length=1, max_length=512)
    prompt_field: str = Field(default="prompt", min_length=1, max_length=160)
    reference_field: str = Field(default="reference", min_length=1, max_length=160)
    prediction_field: str = Field(default="prediction", min_length=1, max_length=160)
    max_new_tokens: int = Field(default=64, ge=1, le=64, strict=True)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)
    device: Literal["auto", "cpu", "cuda", "rocm", "mps", "xpu"] = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"

    @field_validator("model_path", "dataset_path", "output_path", "prompt_field", "reference_field", "prediction_field")
    @classmethod
    def required_values_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prediction job paths and field names must contain non-whitespace text.")
        return value

    @field_validator("temperature", mode="before")
    @classmethod
    def temperature_must_be_numeric_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Temperature must be a finite number between 0 and 2.")
        return value

    @field_validator("adapter_path")
    @classmethod
    def adapter_path_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Adapter path must be omitted or contain non-whitespace text.")
        return value


@router.post("", status_code=202)
def create_prediction_job(request: PredictionJobStartRequest) -> dict[str, Any]:
    """Start a bounded local JSONL prediction job and return its persisted job ID."""
    try:
        job = start_prediction_job(
            active_workspace_root(),
            model_path=request.model_path,
            adapter_path=request.adapter_path,
            dataset_path=request.dataset_path,
            output_path=request.output_path,
            prompt_field=request.prompt_field,
            reference_field=request.reference_field,
            prediction_field=request.prediction_field,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            seed=request.seed,
            device=request.device,
            precision=request.precision,
        )
    except ValueError as exc:
        message = str(exc)
        status = 409 if "compute job is active" in message or "compute job lock" in message else 422
        raise HTTPException(status_code=status, detail=message) from None
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="AXIOM could not start the local prediction worker safely.") from exc

    return {
        "schema_version": 1,
        "job": job,
        "limits": {
            "rows": 100,
            "input_bytes": 48 * 1024 * 1024,
            "input_line_bytes": 768 * 1024,
            "prompt_characters": 8000,
            "generated_tokens_per_row": 64,
            "batch_rows": 4,
            "prediction_characters_per_row": 16000,
            "output_bytes": 64 * 1024 * 1024,
            "output_line_bytes": 1024 * 1024,
        },
        "scope": {
            "network_accessed": False,
            "model_downloaded": False,
            "prompts_returned": False,
            "predictions_returned": False,
            "logs_returned": False,
            "job_state_persisted": True,
            "output_published_only_after_completion": True,
        },
    }


@router.get("")
def list_prediction_job_history(
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    """List bounded local prediction-job summaries for the active workspace."""
    try:
        jobs = list_prediction_jobs(active_workspace_root(), limit=limit)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Prediction job history is unavailable.") from exc
    return {
        "schema_version": 1,
        "jobs": jobs,
        "limit": limit,
        "scope": {
            "prompts_returned": False,
            "predictions_returned": False,
            "logs_returned": False,
            "absolute_paths_returned": False,
            "job_state_persisted": True,
        },
    }


@router.get("/{job_id}")
def get_prediction_job_status(job_id: str) -> dict[str, Any]:
    """Read one safe job summary, recovering only verified completed publications."""
    try:
        job = get_prediction_job(job_id, active_workspace_root())
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Prediction job state could not be read safely.") from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Prediction job was not found in this workspace.")
    return {"schema_version": 1, "job": job}


@router.post("/{job_id}/resume", status_code=202)
def resume_interrupted_prediction_job(job_id: str) -> dict[str, Any]:
    """Resume an interrupted job from its integrity-checked committed output prefix."""
    try:
        job = resume_prediction_job(job_id, active_workspace_root())
    except ValueError as exc:
        message = str(exc)
        status = 409 if (
            "another axiom local compute job is active" in message.casefold()
            or "a local training job is active" in message.casefold()
        ) else 422
        raise HTTPException(status_code=status, detail=message) from None
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Prediction job could not be resumed safely.") from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Prediction job was not found in this workspace.")
    return {
        "schema_version": 1,
        "job": job,
        "scope": {
            "dataset_fingerprint_verified": True,
            "model_and_adapter_fingerprints_verified": True,
            "checkpoint_identity_and_digest_verified": True,
            "partial_output_published": False,
            "prompts_returned": False,
            "predictions_returned": False,
            "generation_resumed_after_restart": True,
        },
    }


@router.post("/{job_id}/cancel")
def request_prediction_job_cancellation(job_id: str) -> dict[str, Any]:
    """Request cooperative cancellation of a job owned by this API process."""
    try:
        result = cancel_prediction_job(job_id, active_workspace_root())
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Prediction job cancellation could not be requested safely.") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction job was not found in this workspace.")
    return {"schema_version": 1, **result}
