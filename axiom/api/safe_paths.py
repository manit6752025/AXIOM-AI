from __future__ import annotations

from pathlib import Path


def safe_child(base: str | Path, user_path: str, *, must_exist: bool = False) -> Path:
    root = Path(base).expanduser().resolve()
    candidate = (root / user_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes the configured base directory")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate
