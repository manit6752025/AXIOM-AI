from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    cpu: str
    ram_gb: float
    gpu: str | None
    vram_gb: float | None


@dataclass(frozen=True)
class OptimizationRequest:
    model: str
    task: str
    priority: str
    target_tps: float | None
    context_length: int


def detect_hardware() -> HardwareProfile:
    ram_gb = 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        ram_gb = pages * size / (1024**3)
    except (AttributeError, OSError, ValueError):
        __import__("logging").getLogger(__name__).debug(
            "intentionally ignored exception", exc_info=True
        )
    return HardwareProfile(
        cpu=platform.processor() or platform.machine(),
        ram_gb=round(ram_gb, 2),
        gpu=None,
        vram_gb=None,
    )


def choose_quantization(model_size_gb: float, hardware: HardwareProfile) -> str:
    available = hardware.ram_gb * 0.75
    if hardware.vram_gb:
        available = max(available, hardware.vram_gb * 0.9)
    if model_size_gb <= available:
        return "FP16"
    if model_size_gb * 0.65 <= available:
        return "Q8"
    if model_size_gb * 0.5 <= available:
        return "Q6_K"
    if model_size_gb * 0.35 <= available:
        return "Q4_K_M"
    return "Q3_K_M"


def build_plan(request: OptimizationRequest, model_size_gb: float) -> dict[str, Any]:
    hardware = detect_hardware()
    quant = choose_quantization(model_size_gb, hardware)
    return {
        "request": asdict(request),
        "hardware": asdict(hardware),
        "model_size_gb": model_size_gb,
        "recommended_quantization": quant,
        "benchmark_required": True,
        "measured_tokens_per_second": None,
    }
