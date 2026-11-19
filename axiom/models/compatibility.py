from __future__ import annotations

import re
from typing import Any

from axiom.core.hardware import HardwareInfo
from axiom.models.registry import Model


_PARAMETER_VALUE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT]?)\s*$", re.IGNORECASE)


def _parameter_count_billions(value: str | None) -> tuple[float | None, str | None, bool]:
    if not value:
        return None, None, False
    match = _PARAMETER_VALUE.fullmatch(value.replace(",", ""))
    if not match:
        return None, None, False
    amount = float(match.group(1))
    suffix = match.group(2).upper()
    if amount <= 0:
        return None, None, False
    if suffix:
        scale = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}[suffix]
        return amount * scale, "unit_suffix", False
    if amount >= 1_000_000:
        return amount / 1_000_000_000, "raw_parameter_count", False
    # Small suffix-free values in this UI convention are treated as billions,
    # but are flagged so the user can correct ambiguous registry metadata.
    return amount, "assumed_billions_without_suffix", True


def _bits_per_parameter(quantization: str | None) -> tuple[float, float] | None:
    value = (quantization or "").strip().casefold().replace(" ", "")
    if not value:
        return None
    if value in {"fp32", "f32", "float32", "32-bit", "int32"}:
        return 32.0, 34.0
    if value in {"fp16", "f16", "float16", "bf16", "bfloat16", "16-bit"}:
        return 16.0, 18.0
    match = re.match(r"^(?:q)?([234568])(?:\b|[_-]|$)", value)
    if not match:
        match = re.match(r"^(?:int)?([234568])-?bit$", value)
    if not match:
        return None
    bits = float(match.group(1))
    return bits, bits + (1.0 if bits >= 3 else 0.8)


def _memory_fit(
    required: dict[str, float] | None,
    physical_gb: float | None,
    reserve_percent: float,
) -> dict[str, Any]:
    if physical_gb is None or physical_gb <= 0:
        return {
            "status": "unknown",
            "physical_gb": None,
            "budget_gb": None,
            "reserve_percent": reserve_percent,
            "required_gb": required,
        }
    budget = round(physical_gb * (1.0 - reserve_percent / 100.0), 2)
    if required is None:
        status = "unknown"
    elif required["high"] <= budget:
        status = "likely"
    elif required["low"] <= budget:
        status = "tight"
    else:
        status = "unlikely"
    return {
        "status": status,
        "physical_gb": round(physical_gb, 2),
        "budget_gb": budget,
        "reserve_percent": reserve_percent,
        "required_gb": required,
    }


def _range(low: float, high: float) -> dict[str, float]:
    return {"low": round(low, 2), "high": round(high, 2)}


def estimate_model_compatibility(model: Model, hardware: HardwareInfo) -> dict[str, Any]:
    """Return evidence-labelled memory estimates and compatibility warnings.

    Estimates intentionally remain screening ranges, not measured runtime needs.
    Current hardware discovery measures NVIDIA VRAM via nvidia-smi only; absence
    of that measurement does not prove that no GPU or unified memory exists.
    """
    parameter_billions, parameter_basis, ambiguous_parameters = _parameter_count_billions(model.parameters)
    bits = _bits_per_parameter(model.quantization)
    estimated_weights: dict[str, float] | None = None
    estimated_runtime: dict[str, float] | None = None
    if parameter_billions is not None and bits is not None:
        estimated_weights = _range(parameter_billions * bits[0] / 8.0, parameter_billions * bits[1] / 8.0)
        # Keep a visible uncertainty range for allocator, metadata, KV cache, and
        # framework overhead. It is a heuristic range, not a confidence interval.
        estimated_runtime = _range(estimated_weights["low"] * 1.1, estimated_weights["high"] * 1.5)

    physical_ram = hardware.ram_gb if isinstance(hardware.ram_gb, (int, float)) and hardware.ram_gb > 0 else None
    physical_vram = hardware.vram_gb if isinstance(hardware.vram_gb, (int, float)) and hardware.vram_gb > 0 else None
    ram_fit = _memory_fit(estimated_runtime, physical_ram, 30.0)
    vram_fit = _memory_fit(estimated_runtime, physical_vram, 15.0)

    model_format = model.format.strip().casefold()
    if "gguf" in model_format:
        format_compatibility = {
            "status": "candidate_not_verified",
            "likely_runtimes": ["Ollama", "llama.cpp"],
            "guidance": "GGUF is commonly consumed by GGUF-capable runtimes; this metadata check does not prove the selected model loads.",
        }
    elif "safetensors" in model_format or model_format in {"pytorch", "pytorch bin", "pt", "bin", "transformers"}:
        format_compatibility = {
            "status": "candidate_not_verified",
            "likely_runtimes": ["Transformers", "AXIOM Engine"],
            "guidance": "Checkpoint compatibility still depends on architecture, config, tokenizer, dtype, dependency versions, and adapter compatibility.",
        }
    else:
        format_compatibility = {
            "status": "unknown",
            "likely_runtimes": [],
            "guidance": "The registered format does not identify a supported runtime family. Inspect the local files and format metadata.",
        }

    warnings: list[str] = [
        "This is a metadata-based screening estimate, not a benchmark or a guarantee that model loading will succeed.",
        "Memory ranges include heuristic overhead; actual needs vary with context length, KV-cache dtype, batch/concurrency, runtime, and GPU layers offloaded.",
        "The registered parameter count and quantization are not independently verified by this report.",
    ]
    if parameter_billions is None:
        warnings.append("Parameter count is missing or unparseable; memory fit cannot be estimated until the registry metadata is corrected.")
    elif ambiguous_parameters:
        warnings.append("Parameter count has no unit suffix and was interpreted as billions; confirm the registered value.")
    if bits is None:
        warnings.append("Quantization/dtype metadata is missing or unsupported; weight and runtime memory ranges are omitted rather than guessed.")
    if physical_vram is None:
        warnings.append("GPU VRAM was not measured by the current NVIDIA nvidia-smi probe. This does not mean a GPU is absent; AMD and Apple unified-memory capacity are not discovered by this probe.")
    if ram_fit["status"] == "tight":
        warnings.append("Estimated runtime memory range crosses the conservative system-RAM budget.")
    elif ram_fit["status"] == "unlikely":
        warnings.append("The lower estimated runtime-memory bound exceeds the conservative system-RAM budget.")
    if vram_fit["status"] == "tight":
        warnings.append("Estimated runtime memory crosses the detected VRAM budget.")
    elif vram_fit["status"] == "unlikely":
        warnings.append("The lower estimated runtime-memory bound exceeds the detected VRAM budget.")

    return {
        "schema_version": 1,
        "model": {
            "id": model.id,
            "name": model.name,
            "format": model.format,
            "parameters": model.parameters,
            "parameter_count_billions": round(parameter_billions, 6) if parameter_billions is not None else None,
            "parameter_basis": parameter_basis,
            "quantization": model.quantization,
            "architecture": model.architecture,
        },
        "hardware": {
            "os_name": hardware.os_name,
            "architecture": hardware.architecture,
            "cpu_cores": hardware.cpu_cores,
            "ram_gb": physical_ram,
            "gpu_name": hardware.gpu_name,
            "vram_gb": physical_vram,
            "vram_probe": "NVIDIA nvidia-smi only",
            "unified_memory_detected": False,
        },
        "quantization_bits_per_parameter": {"low": bits[0], "high": bits[1]} if bits else None,
        "estimated_weight_gb": estimated_weights,
        "estimated_runtime_memory_gb": estimated_runtime,
        "system_ram_fit": ram_fit,
        "gpu_memory_fit": vram_fit,
        "format_compatibility": format_compatibility,
        "warnings": warnings,
        "scope_note": "Estimate only. Runtime compatibility and actual performance require loading and measuring this model with a specific local backend.",
    }
