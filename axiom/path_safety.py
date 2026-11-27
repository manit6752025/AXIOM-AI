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
    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise SafePathError("workspace root must not be a symbolic link")
    try:
        base = supplied_root.resolve(strict=True)
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



def resolve_workspace_output_path(
    raw: str,
    root: Path,
    *,
    allow_existing_output: bool = False,
) -> Path:
    """Resolve a portable workspace output from inspected directory entries.

    Only an absent final leaf may be constructed. Existing components are
    selected by their actual names from a trusted parent directory and
    inspected without following symlinks, junctions, or other reparse points.
    """
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or len(raw) > 4096
        or raw.startswith("/")
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise SafePathError("Use a workspace-relative file path.")

    parts = tuple(raw.split("/"))
    if (
        any(part in {"", ".", ".."} for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
    ):
        raise SafePathError("Use a safe workspace-relative file path.")

    supplied_root = Path(root).absolute()
    current_root = Path(supplied_root.anchor)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for component in supplied_root.parts[1:]:
        current_root = current_root / component
        try:
            root_metadata = current_root.lstat()
        except OSError as exc:
            raise SafePathError("The workspace root is unavailable.") from exc
        is_junction = getattr(current_root, "is_junction", None)
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or bool(reparse_flag and getattr(root_metadata, "st_file_attributes", 0) & reparse_flag)
            or bool(getattr(root_metadata, "st_reparse_tag", 0))
            or (is_junction is not None and is_junction())
        ):
            raise SafePathError("The workspace root must not contain symbolic links or junctions.")

    try:
        workspace = supplied_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SafePathError("The workspace root is unavailable.") from exc
    if not workspace.is_dir():
        raise SafePathError("The workspace root must be a directory.")

    def within_workspace(candidate: Path) -> bool:
        try:
            normalized_candidate = os.path.normcase(
                os.path.normpath(os.path.abspath(os.fspath(candidate)))
            )
            normalized_root = os.path.normcase(
                os.path.normpath(os.path.abspath(os.fspath(workspace)))
            )
            return os.path.commonpath((normalized_candidate, normalized_root)) == normalized_root
        except (OSError, TypeError, ValueError):
            return False

    current = workspace
    for index, part in enumerate(parts):
        final_component = index == len(parts) - 1
        try:
            with os.scandir(current) as entries:
                selected = next((entry for entry in entries if entry.name == part), None)
        except OSError:
            raise SafePathError("A workspace parent directory could not be inspected safely.") from None

        if selected is None:
            if not (final_component and allow_existing_output):
                raise FileNotFoundError("The workspace file was not found.")
            candidate = current / part
            try:
                resolved_parent = current.resolve(strict=True)
            except (OSError, RuntimeError):
                raise SafePathError("The workspace output parent changed during validation.") from None
            if not within_workspace(resolved_parent) or not within_workspace(candidate):
                raise SafePathError("The workspace output must stay inside the workspace.")
            return candidate

        try:
            metadata = selected.stat(follow_symlinks=False)
        except OSError:
            raise SafePathError("A workspace path component changed during validation.") from None

        candidate = Path(selected.path)
        is_junction = getattr(candidate, "is_junction", None)
        is_reparse_point = bool(
            (reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag)
            or getattr(metadata, "st_reparse_tag", 0)
        )
        if selected.is_symlink() or stat.S_ISLNK(metadata.st_mode) or is_reparse_point or (
            is_junction is not None and is_junction()
        ):
            raise SafePathError("Symbolic links and junctions are not supported in workspace paths.")

        if final_component:
            if not stat.S_ISREG(metadata.st_mode):
                raise SafePathError("The workspace output must be a regular file path.")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise SafePathError("Workspace parent directories must already exist.")
        current = candidate

    if current == workspace or not within_workspace(current):
        raise SafePathError("The path must stay inside the workspace.")
    return current

def open_regular_file(path: Path) -> BinaryIO:
    """Open a regular file without following its final or parent symlinks.

    On POSIX, each path component is opened relative to an already-open
    directory descriptor with O_NOFOLLOW. This closes the parent-directory
    swap window left by validating a Path and then calling open(path).
    """
    target = Path(path)
    if ".." in target.parts:
        raise SafePathError("path must not contain parent-directory components")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if (
        os.name == "posix"
        and nofollow
        and getattr(os, "O_DIRECTORY", 0)
        and os.open in getattr(os, "supports_dir_fd", set())
    ):
        absolute = target if target.is_absolute() else Path.cwd() / target
        parts = absolute.parts
        if len(parts) < 2 or parts[0] != absolute.anchor:
            raise SafePathError("path must identify a regular file")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY")
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        directory_fd = os.open(absolute.anchor, directory_flags)
        try:
            for component in parts[1:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            descriptor = os.open(
                parts[-1],
                flags | nofollow | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)
    else:
        # Windows does not expose openat/O_NOFOLLOW consistently. Inspect every
        # path component without following reparse points before opening; also
        # reject junctions and non-directory parents. POSIX keeps the stronger
        # descriptor-relative path above.
        absolute = target if target.is_absolute() else Path.cwd() / target
        current = Path(absolute.anchor)
        components = absolute.parts[1:]
        for index, component in enumerate(components):
            current = current / component
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise SafePathError("path contains a missing or inaccessible component") from exc
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            is_junction = getattr(current, "is_junction", None)
            if (
                stat.S_ISLNK(metadata.st_mode)
                or (reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag)
                or (is_junction is not None and is_junction())
            ):
                raise SafePathError("path must not contain symbolic links or junctions")
            if index < len(components) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise SafePathError("path parents must be regular directories")
        if not components:
            raise SafePathError("path must identify a regular file")
        descriptor = os.open(absolute, flags | nofollow)

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SafePathError("path must refer to a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
