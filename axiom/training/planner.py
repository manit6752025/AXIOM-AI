from __future__ import annotations

import math
from dataclasses import dataclass

from axiom.core.hardware import HardwareInfo


@dataclass
class TrainingPlan:
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


def _estimate_vram(
    parameter_billions: float,
    precision: str,
    batch_size: int,
    sequence_length: int,
) -> float:
    if precision == "4-bit":
        weight_gb_per_billion = 0.5
    elif precision == "8-bit":
        weight_gb_per_billion = 1.0
    else:
        weight_gb_per_billion = 2.0

    weights = parameter_billions * weight_gb_per_billion

    # Conservative planning estimates, not measured runtime usage.
    optimizer = parameter_billions * 0.15
    activations = batch_size * sequence_length * parameter_billions * 0.000012
    runtime = parameter_billions * 0.2

    return round(weights + optimizer + activations + runtime, 2)


def create_training_plan(
    parameter_billions: float,
    hardware: HardwareInfo,
    method: str = "auto",
) -> TrainingPlan:

    if (
        isinstance(parameter_billions, bool)
        or not isinstance(parameter_billions, (int, float))
        or not math.isfinite(parameter_billions)
        or parameter_billions <= 0
    ):
        raise ValueError(
            "Model parameter count must be a finite number greater than zero."
        )

    if method not in {"auto", "qlora", "lora", "full"}:
        raise ValueError("Training method must be one of: auto, qlora, lora, full.")

    vram = hardware.vram_gb

    if vram is not None and (
        isinstance(vram, bool)
        or not isinstance(vram, (int, float))
        or not math.isfinite(vram)
        or vram < 0
    ):
        raise ValueError("Hardware VRAM must be a finite number of GB.")

    if vram is None:
        return TrainingPlan(
            method="development-only",
            precision="CPU",
            lora_rank=8,
            batch_size=1,
            gradient_accumulation=16,
            sequence_length=1024,
            learning_rate=0.0001,
            estimated_vram_gb=None,
            fits_hardware=None,
            reason=(
                "GPU memory could not be read on this machine, so hardware fit "
                "is unknown. Check the model with your local runtime before training."
            ),
        )

    if method == "full":
        selected_method = "full"
    elif method == "lora":
        selected_method = "lora"
    elif method == "qlora":
        selected_method = "qlora"
    else:
        # Default toward QLoRA because it gives useful fine-tuning
        # capability with substantially lower VRAM requirements.
        selected_method = "qlora"

    if selected_method == "full":
        precision = "FP16"
        lora_rank = 0
    elif selected_method == "lora":
        precision = "FP16"
        lora_rank = 16
    else:
        precision = "4-bit"
        lora_rank = 16

    if vram >= 48:
        batch_size = 4
        gradient_accumulation = 4
        sequence_length = 8192
    elif vram >= 24:
        batch_size = 2
        gradient_accumulation = 8
        sequence_length = 4096
    elif vram >= 16:
        batch_size = 1
        gradient_accumulation = 16
        sequence_length = 4096
    elif vram >= 8:
        batch_size = 1
        gradient_accumulation = 32
        sequence_length = 2048
    else:
        batch_size = 1
        gradient_accumulation = 32
        sequence_length = 1024

    estimated = _estimate_vram(
        parameter_billions,
        precision,
        batch_size,
        sequence_length,
    )

    fits = estimated <= vram

    if not fits and selected_method != "qlora":
        selected_method = "qlora"
        precision = "4-bit"
        lora_rank = 16

        estimated = _estimate_vram(
            parameter_billions,
            precision,
            batch_size,
            sequence_length,
        )

        fits = estimated <= vram

    if fits:
        reason = "Configuration is estimated to fit available GPU memory."
    else:
        reason = (
            "Estimated memory exceeds available VRAM. "
            "Reduce sequence length, use gradient accumulation, "
            "or use a smaller/quantized model."
        )

    return TrainingPlan(
        method=selected_method,
        precision=precision,
        lora_rank=lora_rank,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        sequence_length=sequence_length,
        learning_rate=0.0002 if selected_method == "qlora" else 0.0001,
        estimated_vram_gb=estimated,
        fits_hardware=fits,
        reason=reason,
    )
