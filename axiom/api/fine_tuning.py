"""HTTP API for opt-in local LoRA fine-tuning jobs."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from axiom.core.fine_tuning import (
    ActiveFineTuningJobError,
    FineTuningError,
    cancel_finetuning_job,
    get_job,
    list_jobs,
    runtime_status,
    start_finetuning_job,
)

router = APIRouter(prefix="/api/fine-tuning", tags=["local-fine-tuning"])


class StartFineTuningRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=4096)
    dataset_path: str = Field(min_length=1, max_length=4096)
    output_dir: str = Field(min_length=1, max_length=4096)
    text_field: str = Field(default="text", min_length=1, max_length=64)
    max_steps: int = Field(default=20, ge=1, le=1000)
    sequence_length: int = Field(default=512, ge=32, le=2048)
    learning_rate: float = Field(default=0.0001, ge=0.000001, le=0.01)
    lora_rank: int = Field(default=8, ge=1, le=64)


@router.get("/runtime")
def fine_tuning_runtime() -> dict[str, Any]:
    return runtime_status()


@router.get("/jobs")
def fine_tuning_jobs(limit: int = 20) -> dict[str, Any]:
    try:
        return {"jobs": list_jobs(limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.get("/jobs/{job_id}")
def fine_tuning_job(job_id: str) -> dict[str, Any]:
    try:
        job = get_job(job_id)
    except (ValueError, FineTuningError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job was not found.")
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_fine_tuning(job_id: str) -> dict[str, Any]:
    try:
        job = cancel_finetuning_job(job_id)
    except (ValueError, FineTuningError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job was not found.")
    return job


@router.post("/jobs", status_code=202)
def start_fine_tuning(request: StartFineTuningRequest) -> dict[str, Any]:
    try:
        return start_finetuning_job(request.model_dump())
    except ActiveFineTuningJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except FineTuningError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
