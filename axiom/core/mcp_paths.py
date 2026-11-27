"""Descriptor-relative workspace paths for MCP file tools."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO

import axiom.path_safety as path_safety


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
        and os.scandir in os.supports_fd
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
    )


def _is_link_like(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_reparse_tag", 0)
        or (reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag)
    ):
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
    """Delegate portable MCP path resolution to the shared safe-path boundary."""
    return path_safety.resolve_workspace_output_path(
        raw,
        root,
        allow_existing_output=allow_existing_output,
    )

def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        normalized_path = os.path.normcase(os.path.normpath(os.fspath(path)))
        normalized_root = os.path.normcase(os.path.normpath(os.fspath(root)))
        return os.path.commonpath((normalized_path, normalized_root)) == normalized_root
    except (OSError, TypeError, ValueError):
        return False


def _find_directory_entry(
    directory_descriptor: int,
    requested_name: str,
) -> tuple[str, os.stat_result, bool] | None:
    """Return a matching entry name sourced from a pinned directory listing.

    The requested name is only compared with directory entries. It is never
    passed to a filesystem path API, so user-controlled text cannot become a
    path operand after this boundary.
    """
    try:
        with os.scandir(directory_descriptor) as entries:
            selected = next((entry for entry in entries if entry.name == requested_name), None)
            if selected is None:
                return None
            metadata = selected.stat(follow_symlinks=False)
            is_link_like = (
                selected.is_symlink()
                or stat.S_ISLNK(metadata.st_mode)
                or bool(
                    getattr(metadata, "st_reparse_tag", 0)
                    or (
                        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                        and getattr(metadata, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    )
                )
            )
            return selected.name, metadata, is_link_like
    except OSError as exc:
        raise ValueError("A workspace directory could not be inspected safely.") from exc


def _open_directory_entry(
    directory_descriptor: int,
    requested_name: str,
    directory_flags: int,
) -> int:
    selected = _find_directory_entry(directory_descriptor, requested_name)
    if selected is None:
        raise ValueError("A workspace parent directory is missing or unsafe.")
    name, expected, is_link_like = selected
    if is_link_like or not stat.S_ISDIR(expected.st_mode):
        raise ValueError("Workspace parent directories must be real directories.")

    descriptor = os.open(name, directory_flags, dir_fd=directory_descriptor)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("A workspace directory changed while it was being opened.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_file_entry(
    directory_descriptor: int,
    requested_name: str,
    flags: int,
) -> tuple[str, int]:
    selected = _find_directory_entry(directory_descriptor, requested_name)
    if selected is None:
        raise ValueError("The workspace file is missing or unsafe.")
    name, expected, is_link_like = selected
    if is_link_like or not stat.S_ISREG(expected.st_mode):
        raise ValueError("The workspace path must refer to a regular file.")

    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("The workspace file changed while it was being opened.")
        return name, descriptor
    except BaseException:
        os.close(descriptor)
        raise


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
            next_descriptor = _open_directory_entry(descriptor, part, directory_flags)
            os.close(descriptor)
            descriptor = next_descriptor
    except (OSError, ValueError) as exc:
        os.close(descriptor)
        raise ValueError("A workspace parent directory is missing or unsafe.") from exc

    relative = Path(*parts)
    return workspace / relative, descriptor, parts[-1]


def open_mcp_workspace_file(raw: str, root: Path) -> tuple[Path, BinaryIO]:
    """Open a regular workspace file while pinning every path component."""
    parts = _parts(raw)
    if not _descriptor_access_supported():
        workspace = _workspace_root(root)
        path = _fallback_workspace_path(raw, workspace)
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(workspace)
            stream = path_safety.open_regular_file(resolved_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("The workspace file is missing or unsafe.") from exc
        return resolved_path, stream
        path = _fallback_workspace_path(raw, root)
        try:
            stream = path_safety.open_regular_file(path)
        except (OSError, ValueError) as exc:
            raise ValueError("The workspace file is missing or unsafe.") from exc
        return path, stream

    path, parent_descriptor, name = _open_parent(root, parts)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        _selected_name, descriptor = _open_regular_file_entry(
            parent_descriptor,
            name,
            flags,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("The workspace file is missing or unsafe.") from exc
    finally:
        os.close(parent_descriptor)
    return path, os.fdopen(descriptor, "rb")


def open_mcp_workspace_output_parent(
    raw: str,
    root: Path,
) -> tuple[Path, int | None, str, bool]:
    """Pin an existing parent directory and report whether the output name is free."""
    if not _descriptor_access_supported() or (
        os.link not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd
    ):
        path, available = path_safety.resolve_workspace_output_path(
            raw,
            root,
            allow_existing_output=True,
            return_availability=True,
        )
        return path, None, path.name, available

    parts = _parts(raw)
    path, parent_descriptor, name = _open_parent(root, parts)
    try:
        existing_entry = _find_directory_entry(parent_descriptor, name)
        if existing_entry is None:
            available = True
        else:
            _existing_name, output_stat, is_link_like = existing_entry
            if is_link_like or not stat.S_ISREG(output_stat.st_mode):
                raise ValueError("The workspace output must be a regular file path.")
            available = False
    except (OSError, ValueError) as exc:
        os.close(parent_descriptor)
        raise ValueError("The workspace output path could not be inspected safely.") from exc
    return path, parent_descriptor, name, available
