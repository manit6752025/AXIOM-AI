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


def _workspace_root(root: Path) -> Path:
    """Resolve a real workspace directory without accepting a symlink/junction root."""
    supplied = Path(root)
    if _is_link_like(supplied):
        raise ValueError("The MCP workspace root must not be a symbolic link.")
    try:
        workspace = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("The MCP workspace root is unavailable.") from exc
    if not workspace.is_dir():
        raise ValueError("The MCP workspace root must be a directory.")
    return workspace


def _fallback_workspace_path(
    raw: str,
    root: Path,
    *,
    allow_existing_output: bool = False,
) -> Path:
    """Resolve portable paths by matching actual directory entries, not joins.

    The fallback is used on platforms without descriptor-relative no-follow
    APIs. Existing components are selected from parent directory entries and
    inspected without following links. Only an absent final output leaf is
    joined to the already-validated parent.
    """
    parts = _parts(raw)
    workspace = _workspace_root(root)
    current = workspace
    for index, part in enumerate(parts):
        is_final = index == len(parts) - 1
        try:
            with os.scandir(current) as entries:
                selected = next((entry for entry in entries if entry.name == part), None)
        except OSError:
            raise ValueError("A workspace parent directory could not be inspected safely.") from None

        if selected is None:
            if not (is_final and allow_existing_output):
                raise ValueError("The workspace file was not found or is not a regular file.")
            # part has already passed _parts(): one non-empty component with
            # separators, dot segments and drive prefixes rejected.
            parent = current
            candidate = parent / part
            try:
                resolved_parent = parent.resolve(strict=True)
            except (OSError, RuntimeError):
                raise ValueError("The workspace output parent changed while being inspected.") from None
            if not _path_is_within_root(resolved_parent, workspace):
                raise ValueError("The path must stay inside the MCP workspace.")
            if candidate.exists() or _is_link_like(candidate):
                raise ValueError("The workspace output path changed while being inspected.")
            return candidate

        try:
            metadata = selected.stat(follow_symlinks=False)
        except OSError:
            raise ValueError("A workspace path component changed while being inspected.") from None
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        is_reparse_point = bool(
            reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag
        )
        candidate = Path(selected.path)
        if stat.S_ISLNK(metadata.st_mode) or is_reparse_point or _is_link_like(candidate):
            raise ValueError("Symbolic links and junctions are not supported in MCP file paths.")

        if is_final:
            if allow_existing_output:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("The workspace output must be a regular file path.")
            elif not stat.S_ISREG(metadata.st_mode):
                raise ValueError("The workspace file was not found or is not a regular file.")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Workspace parent directories must already exist.")
        current = candidate

    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The workspace path changed while being inspected.") from None
    if not _path_is_within_root(resolved, workspace) or resolved == workspace:
        raise ValueError("The path must stay inside the MCP workspace.")
    return resolved


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        normalized_path = os.path.normcase(os.path.normpath(os.fspath(path)))
        normalized_root = os.path.normcase(os.path.normpath(os.fspath(root)))
        return os.path.commonpath((normalized_path, normalized_root)) == normalized_root
    except (OSError, TypeError, ValueError):
        return False


def _open_parent(root: Path, parts: tuple[str, ...]) -> tuple[Path, int, str]:
    # Preserve the explicit symlink/junction-root rejection rather than
    # collapsing it into the generic descriptor-open failure below.
    workspace = _workspace_root(root)
    descriptor: int | None = None
    try:
        process_root = Path.cwd().resolve(strict=True)
        workspace_parts = workspace.relative_to(process_root).parts
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(".", directory_flags)
        for part in workspace_parts:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
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
