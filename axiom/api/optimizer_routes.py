from __future__ import annotations

import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from axiom.api.model_paths import validate_model_path

router = APIRouter(prefix="/api/optimization", tags=["optimization"])

_SAFE_ROOTS = tuple(
    # codeql[py/path-injection]
    Path(validate_model_path(p)).resolve()
    for p in (
        os.environ.get("AXIOM_MODEL_ROOT", "./models"),
        os.environ.get("AXIOM_DATA_ROOT", "./data"),
    )
)


def _safe_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    # codeql[py/path-injection]
    candidate = Path(validate_model_path(raw)).expanduser().resolve()
    if any(candidate == root or root in candidate.parents for root in _SAFE_ROOTS):
        return candidate
    raise HTTPException(
        status_code=400, detail="Path must be inside an AXIOM model/data root"
    )


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return None


def _gpu_info() -> tuple[str, float | None]:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return "No GPU reported", None
    try:
        import subprocess

        out = (
            subprocess.check_output(
                [
                    nvidia,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()[0]
        )
        name, memory = [part.strip() for part in out.split(",", 1)]
        return name, round(float(memory) / 1024, 1)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "NVIDIA GPU detected", None


class PlanRequest(BaseModel):
    model: str = Field(min_length=1, max_length=512)
    goal: str = Field(min_length=1, max_length=64)
    priority: str = Field(min_length=1, max_length=32)
    target_tps: float | None = Field(default=None, ge=0, le=100000)
    context_length: int = Field(default=8192, ge=256, le=131072)
    dataset_path: str | None = Field(default=None, max_length=512)


@router.get("/hardware")
def hardware() -> dict[str, Any]:
    gpu_name, vram = _gpu_info()
    return {
        "cpu_name": platform.processor() or platform.machine(),
        "ram_gb": _ram_gb(),
        "gpu_name": gpu_name,
        "vram_gb": vram,
    }


@router.post("/plan")
def plan(request: PlanRequest) -> dict[str, Any]:
    model_path = None
    if (
        request.model.startswith("/")
        or request.model.startswith("~")
        or "/" in request.model
        or "\\" in request.model
    ):
        model_path = _safe_path(request.model)
        # codeql[py/path-injection]
        if model_path is not None and not model_path.exists():
            raise HTTPException(status_code=404, detail="Model path does not exist")

    ram = _ram_gb() or 8.0
    _, vram = _gpu_info()
    memory = vram or ram

    params_b = None
    match = re.search(
        r"(?:^|[-_/])([0-9]+(?:\.[0-9]+)?)b(?:$|[-_/])", request.model.lower()
    )
    if match:
        params_b = float(match.group(1))
    # codeql[py/path-injection]
    elif model_path and model_path.is_file():
        params_b = max(0.5, model_path.stat().st_size * 8 / 1e9)

    if params_b is None:
        quant = "Auto / benchmark candidates"
    elif params_b * 2 <= memory * 0.85:
        quant = "FP16 / BF16 candidate"
    elif params_b <= memory * 0.95:
        quant = "Q8 candidate"
    elif params_b * 0.55 <= memory * 0.95:
        quant = "Q5_K_M candidate"
    else:
        quant = "Q4_K_M candidate"

    dataset_action = (
        "Analyze dataset relevance before retraining"
        if request.dataset_path
        else "Not requested"
    )
    if request.priority == "speed":
        dataset_action = (
            "Keep only task-relevant examples; prioritize shorter outputs"
            if request.dataset_path
            else dataset_action
        )
    elif request.priority == "quality":
        dataset_action = (
            "Filter duplicates/noise while preserving coverage"
            if request.dataset_path
            else dataset_action
        )

    return {
        "model": request.model,
        "goal": request.goal,
        "priority": request.priority,
        "target_tps": request.target_tps,
        "context_length": request.context_length,
        "quantization": quant,
        "dataset_action": dataset_action,
        "benchmark_required": True,
        "measured_tps": None,
        "model_parameters_b": params_b,
        "memory_budget_gb": round(memory, 1),
    }
