from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    cpu: str
    cores: int
    ram_gb: float
    gpu: str | None
    vram_gb: float | None


def detect_hardware() -> HardwareProfile:
    ram_gb = 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        ram_gb = round((pages * size) / (1024**3), 2)
    except (AttributeError, OSError, ValueError):
        __import__("logging").getLogger(__name__).debug(
            "intentionally ignored exception", exc_info=True
        )
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = "NVIDIA"
    elif shutil.which("rocminfo"):
        gpu = "AMD"
    return HardwareProfile(
        platform.processor() or platform.machine(),
        os.cpu_count() or 1,
        ram_gb,
        gpu,
        None,
    )
