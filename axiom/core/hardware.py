from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    os_name: str
    architecture: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str | None
    vram_gb: float | None
    cuda_available: bool


def _detect_ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024**3), 2)
    except (ValueError, OSError):
        return 0.0


def _detect_nvidia() -> tuple[str | None, float | None, bool]:
    if shutil.which("nvidia-smi") is None:
        return None, None, False

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        line = result.stdout.strip().splitlines()[0]
        name, memory = [part.strip() for part in line.split(",", 1)]

        return name, round(float(memory) / 1024, 2), True

    except (
        subprocess.SubprocessError,
        ValueError,
        IndexError,
    ):
        return None, None, False


def detect_hardware() -> HardwareInfo:
    gpu_name, vram_gb, cuda = _detect_nvidia()

    return HardwareInfo(
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=_detect_ram_gb(),
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cuda_available=cuda,
    )


def estimate_model_fit(
    parameter_billions: float,
    hardware: HardwareInfo,
) -> str:
    if (
        isinstance(parameter_billions, bool)
        or not isinstance(parameter_billions, (int, float))
        or not math.isfinite(parameter_billions)
        or parameter_billions <= 0
    ):
        raise ValueError(
            "Model parameter count must be a finite number greater than zero."
        )

    if hardware.vram_gb is None:
        return "⚠ GPU unavailable / unknown"

    if (
        isinstance(hardware.vram_gb, bool)
        or not isinstance(hardware.vram_gb, (int, float))
        or not math.isfinite(hardware.vram_gb)
        or hardware.vram_gb < 0
    ):
        raise ValueError("Hardware VRAM must be a finite number of GB.")

    fp16_gb = parameter_billions * 2 * 1.2
    q4_gb = parameter_billions * 0.5 * 1.2

    if hardware.vram_gb >= fp16_gb:
        return "✓ FP16 / Q4"

    if hardware.vram_gb >= q4_gb:
        return "✓ Q4 quantized"

    return "✗ Insufficient VRAM"
