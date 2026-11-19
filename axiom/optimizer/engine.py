from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class OptimizationRequest:
    model: str
    agent_type: str = "general"
    target_tokens_per_second: float = 10.0
    context_length: int = 4096
    quality: str = "balanced"
    output_dir: str = "optimized-model"
    quantization: str = "auto"


@dataclass
class OptimizationResult:
    model: str
    hardware: dict[str, Any]
    recommendation: dict[str, Any]
    commands: list[str]
    output_dir: str
    status: str = "planned"


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def detect_hardware() -> dict[str, Any]:
    """Best-effort local hardware detection; never fails the optimizer."""
    hw: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": None,
        "gpu": None,
        "gpu_vram_gb": None,
        "gpu_backend": None,
    }
    try:
        if Path("/proc/meminfo").exists():
            kb = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
            hw["ram_gb"] = round(kb / 1024 / 1024, 2)
    except Exception:  # noqa: BLE001
        __import__("logging").getLogger(__name__).debug(
            "intentionally ignored exception", exc_info=True
        )

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        name = _cmd(nvidia, "--query-gpu=name", "--format=csv,noheader")
        vram = _cmd(nvidia, "--query-gpu=memory.total", "--format=csv,noheader,nounits")
        if name:
            hw["gpu"] = name.splitlines()[0].strip()
            hw["gpu_backend"] = "cuda"
        if vram:
            try:
                hw["gpu_vram_gb"] = round(float(vram.splitlines()[0].strip()) / 1024, 2)
            except ValueError:
                __import__("logging").getLogger(__name__).debug(
                    "intentionally ignored exception", exc_info=True
                )

    if not hw["gpu"] and shutil.which("rocminfo"):
        hw["gpu_backend"] = "rocm"
        info = _cmd("rocminfo") or ""
        for line in info.splitlines():
            if "Marketing Name:" in line:
                hw["gpu"] = line.split(":", 1)[1].strip()
                break

    if (
        not hw["gpu"]
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
    ):
        hw["gpu"] = "Apple Silicon"
        hw["gpu_backend"] = "metal"

    return hw


def _model_size_hint(model: str) -> float | None:
    """Infer parameter count from common model IDs such as 7B/8B/14B."""
    import re

    m = re.search(r"(?:^|[-_/])([0-9]+(?:\.[0-9]+)?)B(?:$|[-_/])", model, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _choose_quantization(
    params_b: float | None, hw: dict[str, Any], quality: str
) -> str:
    vram = hw.get("gpu_vram_gb")
    ram = hw.get("ram_gb") or 8
    if params_b is None:
        return "int4" if quality != "quality" else "int8"
    # Approximate weights only. Runtime KV/cache overhead is intentionally not hidden.
    if vram and params_b * 2.0 <= vram * 0.80:
        return "fp16"
    if vram and params_b * 1.0 <= vram * 0.80:
        return "int8"
    if params_b * 0.55 <= ram * 0.70:
        return "int4"
    return "int4-cpu"


def optimize_model(req: OptimizationRequest) -> OptimizationResult:
    hw = detect_hardware()
    params_b = _model_size_hint(req.model)
    quant = (
        req.quantization
        if req.quantization != "auto"
        else _choose_quantization(params_b, hw, req.quality)
    )

    # Agent type changes the optimization profile rather than deleting arbitrary model layers.
    profiles = {
        "chat": {
            "keep": ["conversation", "instruction following"],
            "sampling": "stable",
        },
        "coding": {"keep": ["code generation", "long context"], "sampling": "precise"},
        "reasoning": {
            "keep": ["reasoning", "instruction following"],
            "sampling": "conservative",
        },
        "tool-use": {
            "keep": ["tool calling", "structured output"],
            "sampling": "deterministic",
        },
        "general": {"keep": ["general instruction following"], "sampling": "balanced"},
    }
    profile = profiles.get(req.agent_type, profiles["general"])

    commands: list[str] = []
    if req.model.startswith("http://") or req.model.startswith("https://"):
        raise ValueError("Use a Hugging Face model ID or local path, not a direct URL.")

    # Prefer llama.cpp conversion when installed; otherwise a local CLI workflow
    # can install or select the backend later.
    commands.append(
        f"huggingface-cli download {req.model} --local-dir {req.output_dir}/source"
    )
    if quant in {"int4", "int4-cpu"}:
        commands.append(
            f"# Quantize to 4-bit ({quant}) using the selected compatible backend"
        )
    elif quant == "int8":
        commands.append("# Quantize to 8-bit using the selected compatible backend")
    else:
        commands.append(
            "# Keep FP16/BF16 weights; optimize runtime settings for detected accelerator"
        )

    return OptimizationResult(
        model=req.model,
        hardware=hw,
        recommendation={
            "agent_type": req.agent_type,
            "quantization": quant,
            "target_tokens_per_second": req.target_tokens_per_second,
            "context_length": req.context_length,
            "quality": req.quality,
            "parameter_billions_hint": params_b,
            "profile": profile,
            "note": "Performance is estimated; AXIOM should benchmark the produced artifact before claiming the target is met.",
        },
        commands=commands,
        output_dir=req.output_dir,
    )


def save_result(result: OptimizationResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2) + "\n")
