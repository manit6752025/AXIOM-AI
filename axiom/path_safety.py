"""Safe selection of existing workspace filesystem entries."""

from __future__ import annotations

import os
import stat
from collections.abc import Collection
from pathlib import Path
from typing import BinaryIO


DEFAULT_SKIP_DIRECTORIES = frozenset({
    ".git", ".venv", "node_modules", "__pycache__",
    "dist", "build", ".axiom",
})


class SafePathError(ValueError):
    """A client supplied an unsafe or unsupported relative path."""


def _parts(raw: str) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw or not raw.strip():
        raise SafePathError("path is required")
    if len(raw) > 4096:
        raise SafePathError("path is too long")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise SafePathError("path contains a control character")
    if "\\" in raw or raw.startswith("/"):
        raise SafePathError("use a workspace-relative path")

    while raw.startswith("./"):
        raw = raw[2:]

    parts = tuple(raw.split("/"))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise SafePathError("path contains an unsafe component")
    if len(parts[0]) >= 2 and parts[0][1] == ":":
        raise SafePathError("absolute paths are not supported")
    return parts


def find_existing_relative_path(
    root: Path,
    raw: str,
    *,
    allow_directories: bool = True,
    skip_directories: Collection[str] = DEFAULT_SKIP_DIRECTORIES,
) -> Path:
    """Find an existing entry using trusted directory-entry names.

    The request is never joined to the root to produce a filesystem path.
    Symlink entries and symlinked directories are rejected or skipped.
    """
    parts = _parts(raw)
    try:
        base = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError("Workspace root is unavailable") from exc

    if not base.is_dir():
        raise FileNotFoundError("Workspace root is unavailable")

    parent_key = "/".join(parts[:-1]) or "."
    wanted_name = parts[-1]

    for current, directories, filenames in os.walk(base, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories
            if name not in skip_directories
            and not (current_path / name).is_symlink()
        )

        current_key = (
            "."
            if current_path == base
            else current_path.relative_to(base).as_posix()
        )
        if current_key != parent_key:
            continue

        for name in sorted(directories + filenames):
            if name != wanted_name:
                continue

            candidate = current_path / name
            if candidate.is_symlink():
                raise SafePathError("symlink paths are not supported")

            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise FileNotFoundError("Requested entry is unavailable") from exc

            if resolved == base or not resolved.is_relative_to(base):
                raise SafePathError("path must stay inside the workspace")
            if not (resolved.is_file() or resolved.is_dir()):
                raise SafePathError("only regular files and directories are supported")
            if not allow_directories and not resolved.is_file():
                raise SafePathError("path must refer to a regular file")

            return resolved

        raise FileNotFoundError("Requested workspace entry was not found")

    raise FileNotFoundError("Requested workspace entry was not found")


def open_regular_file(path: Path) -> BinaryIO:
    """Open a previously selected regular file without following its final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SafePathError("path must refer to a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
