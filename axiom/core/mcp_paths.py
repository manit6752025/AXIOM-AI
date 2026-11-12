"""Descriptor-relative workspace paths for MCP file tools."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO


def _parts(raw: str) -> tuple[str, ...]:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or len(raw) > 4096
        or raw.startswith("/")
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("Use a workspace-relative file path.")

    parts = tuple(raw.split("/"))
    if (
        any(part in {"", ".", ".."} for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
    ):
        raise ValueError("Use a safe workspace-relative file path.")
    return parts


def _descriptor_access_supported() -> bool:
    return (
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
    )


def _is_link_like(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_reparse_tag", 0):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _fallback_workspace_path(
    raw: str,
    root: Path,
    *,
    allow_existing_output: bool = False,
) -> Path:
    parts = _parts(raw)
    workspace = Path(root).resolve(strict=True)
    candidate = workspace
    for index, part in enumerate(parts):
        candidate = candidate / part
        if _is_link_like(candidate):
            raise ValueError("Symbolic links are not supported in MCP file paths.")
        is_final = index == len(parts) - 1
        if is_final and not allow_existing_output and not candidate.is_file():
            raise ValueError("The workspace file was not found or is not a regular file.")
        if not is_final and not candidate.is_dir():
            raise ValueError("Workspace parent directories must already exist.")

    resolved = candidate.resolve(strict=not allow_existing_output or candidate.exists())
    if resolved == workspace or not resolved.is_relative_to(workspace):
        raise ValueError("The path must stay inside the MCP workspace.")
    return resolved


def _open_parent(root: Path, parts: tuple[str, ...]) -> tuple[Path, int, str]:
    descriptor: int | None = None
    try:
        workspace = Path(root).resolve(strict=True)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(".", directory_flags)
        expected = os.stat(workspace)
        opened = os.fstat(descriptor)
        if (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("The MCP workspace changed while opening it.")
    except (OSError, RuntimeError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError("The MCP workspace could not be opened safely.") from exc

    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ValueError("A workspace parent directory is missing or unsafe.") from exc

    relative = Path(*parts)
    return workspace / relative, descriptor, parts[-1]


def open_mcp_workspace_file(raw: str, root: Path) -> tuple[Path, BinaryIO]:
    """Open a regular workspace file while pinning every path component."""
    parts = _parts(raw)
    if not _descriptor_access_supported():
        path = _fallback_workspace_path(raw, root)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError("The workspace file is missing or unsafe.") from exc
        try:
            regular_file = stat.S_ISREG(os.fstat(descriptor).st_mode)
        except OSError as exc:
            os.close(descriptor)
            raise ValueError("The workspace file could not be checked safely.") from exc
        if not regular_file:
            os.close(descriptor)
            raise ValueError("The workspace path must refer to a regular file.")
        return path, os.fdopen(descriptor, "rb")

    path, parent_descriptor, name = _open_parent(root, parts)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        os.close(parent_descriptor)
        raise ValueError("The workspace file is missing or unsafe.") from exc
    os.close(parent_descriptor)

    try:
        regular_file = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except OSError as exc:
        os.close(descriptor)
        raise ValueError("The workspace file could not be checked safely.") from exc
    if not regular_file:
        os.close(descriptor)
        raise ValueError("The workspace path must refer to a regular file.")
    return path, os.fdopen(descriptor, "rb")


def open_mcp_workspace_output_parent(
    raw: str,
    root: Path,
) -> tuple[Path, int | None, str, bool]:
    """Pin an existing parent directory and report whether the output name is free."""
    if not _descriptor_access_supported() or (
        os.link not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd
    ):
        path = _fallback_workspace_path(raw, root, allow_existing_output=True)
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise ValueError("The workspace output must be a regular file path.")
            return path, None, path.name, False
        return path, None, path.name, True

    parts = _parts(raw)
    path, parent_descriptor, name = _open_parent(root, parts)
    try:
        output_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        available = True
    except OSError as exc:
        os.close(parent_descriptor)
        raise ValueError("The workspace output path could not be inspected safely.") from exc
    else:
        if not stat.S_ISREG(output_stat.st_mode):
            os.close(parent_descriptor)
            raise ValueError("The workspace output must be a regular file path.")
        available = False
    return path, parent_descriptor, name, available
