"""Hardware-aware AXIOM Beta.6 optimization planner.

Plans optimization candidates; measured throughput is only reported after a real benchmark.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass


@dataclass
class HardwareProfile:
    cpu: str
    cpu_count: int
    ram_gb: float
    gpu: str | None
    vram_gb: float | None


@dataclass
class OptimizationPlan:
    model: str
    task: str
    priority: str
    target_tps: float | None
    context_length: int
    hardware: HardwareProfile
    quantization_candidates: list[str]
    selected_quantization: str
    estimated_weight_gb: float | None
    benchmark_required: bool = True
    measured_tps: float | None = None


def detect_hardware() -> HardwareProfile:
    ram_gb = 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        ram_gb = pages * page_size / (1024**3)
    except (ValueError, OSError):
        __import__("logging").getLogger(__name__).debug(
            "intentionally ignored exception", exc_info=True
        )
    return HardwareProfile(
        platform.processor() or platform.machine(),
        os.cpu_count() or 1,
        round(ram_gb, 2),
        None,
        None,
    )


def plan(
    model: str,
    task: str = "general",
    priority: str = "balanced",
    target_tps: float | None = None,
    context_length: int = 8192,
    parameter_billion: float | None = None,
) -> dict:
    hw = detect_hardware()
    candidates = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    selected = (
        "Q4_K_M"
        if priority == "speed"
        else ("Q6_K" if priority == "quality" else "Q5_K_M")
    )
    estimated = None
    if parameter_billion:
        bits = {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q6_K": 6.5, "Q8_0": 8.0}[selected]
        estimated = round(parameter_billion * bits / 8, 2)
    return asdict(
        OptimizationPlan(
            model,
            task,
            priority,
            target_tps,
            context_length,
            hw,
            candidates,
            selected,
            estimated,
        )
    )
