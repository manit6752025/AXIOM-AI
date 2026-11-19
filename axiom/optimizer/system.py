from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SystemInfo:
    os: str
    arch: str
    cpu_count: int | None
    ram_gb: float | None
    gpu_available: bool | None
    gpu_name: str | None
    vram_gb: float | None
    disk_free_gb: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "arch": self.arch,
            "cpu_count": self.cpu_count,
            "ram_gb": round(self.ram_gb, 2) if self.ram_gb is not None else None,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "vram_gb": round(self.vram_gb, 2) if self.vram_gb is not None else None,
            "disk_free_gb": round(self.disk_free_gb, 2) if self.disk_free_gb is not None else None,
        }


def _ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size / (1024**3)
    except (AttributeError, OSError, ValueError):
        # Some platforms do not expose these counters; try the Linux source next.
        pass

    path = Path("/proc/meminfo")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        # procfs is unavailable or malformed on this system, so RAM is unknown.
        pass

    return None


def _nvidia() -> tuple[bool | None, str | None, float | None]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None, None, None

    try:
        import subprocess

        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return False, None, None

        first = result.stdout.strip().splitlines()[0]
        name, memory = [x.strip() for x in first.split(",", 1)]
        return True, name, float(memory) / 1024
    except (OSError, ValueError):
        return False, None, None


def inspect_system() -> SystemInfo:
    gpu, gpu_name, vram = _nvidia()

    free_gb = None
    try:
        free_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
    except OSError:
        import logging as _axiom_logging

        _axiom_logging.getLogger(__name__).debug(
            "intentionally ignored exception", exc_info=True
        )

    return SystemInfo(
        os=platform.system(),
        arch=platform.machine(),
        cpu_count=os.cpu_count(),
        ram_gb=_ram_gb(),
        gpu_available=gpu,
        gpu_name=gpu_name,
        vram_gb=vram,
        disk_free_gb=free_gb,
    )
