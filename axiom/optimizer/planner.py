from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareProfile


@dataclass(frozen=True)
class OptimizationPlan:
    quantization: str
    context_length: int
    target_tps: float | None
    benchmark_required: bool = True


def recommend_plan(
    *,
    parameter_count_b: float,
    hardware: HardwareProfile,
    priority: str = "balanced",
    target_tps: float | None = None,
    context_length: int = 8192,
) -> OptimizationPlan:
    # Conservative inference-memory planning. Actual throughput is only established by benchmark.
    if parameter_count_b >= 20 or parameter_count_b >= 8:
        quant = "Q4_K_M"
    elif parameter_count_b >= 3:
        quant = "Q5_K_M" if priority == "quality" else "Q4_K_M"
    else:
        quant = "Q8_0" if priority == "quality" else "Q4_K_M"
    if priority == "speed":
        quant = "Q4_K_M"
        context_length = min(context_length, 8192)
    elif priority == "quality":
        context_length = min(context_length, 16384)
    return OptimizationPlan(quant, max(512, context_length), target_tps, True)
