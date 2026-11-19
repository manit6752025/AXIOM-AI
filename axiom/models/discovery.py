"""Bounded discovery of local model folders without following symlinks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}
_SKIP_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}


def discover_local_model_folders(
    root: str | Path,
    *,
    max_candidates: int = 100,
    max_directories: int = 5000,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Return local model folders under a caller-selected trusted root."""
    base = Path(root)
    if base.is_symlink():
        raise ValueError("Model discovery root must not be a symbolic link.")
    try:
        base = base.resolve(strict=True)
    except FileNotFoundError:
        return {"candidates": [], "truncated": False}
    if not base.is_dir():
        raise ValueError("Model discovery root must be a directory.")
    if not 1 <= max_candidates <= 100 or not 1 <= max_directories <= 5000 or not 0 <= max_depth <= 8:
        raise ValueError("Model discovery limits are outside supported bounds.")

    candidates = []
    visited = 0
    truncated = False
    for current, directories, filenames in os.walk(base, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.relative_to(base).parts)
        if depth >= max_depth:
            if directories:
                truncated = True
            directories.clear()
        else:
            directories[:] = sorted(
                name for name in directories
                if name not in _SKIP_DIRECTORIES
                and not name.startswith(".")
                and not (current_path / name).is_symlink()
            )
        visited += 1
        if visited > max_directories:
            truncated = True
            break

        usable_files = [
            current_path / name
            for name in filenames
            if not (current_path / name).is_symlink()
            and (name == "config.json" or Path(name).suffix.lower() in _WEIGHT_SUFFIXES)
        ]
        if not usable_files:
            continue
        weights = [file for file in usable_files if file.suffix.lower() in _WEIGHT_SUFFIXES]
        try:
            weight_size = sum(file.stat().st_size for file in weights)
        except OSError:
            continue
        relative_path = current_path.relative_to(base).as_posix()
        candidates.append({
            "path": relative_path,
            "format": (
                "GGUF" if any(file.suffix.lower() == ".gguf" for file in weights)
                else "Safetensors" if any(file.suffix.lower() == ".safetensors" for file in weights)
                else "PyTorch" if weights
                else "Unknown"
            ),
            "has_config": any(file.name == "config.json" for file in usable_files),
            "weight_file_count": len(weights),
            "weight_size_bytes": weight_size,
        })
        if len(candidates) >= max_candidates:
            truncated = True
            break

    return {"candidates": candidates, "truncated": truncated}
