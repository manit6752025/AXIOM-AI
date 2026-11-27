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
    return_availability: bool = False,
) -> Path | tuple[Path, bool]:
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
                entries_by_name = {entry.name: entry for entry in entries}
                selected = entries_by_name.get(part)
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
            return (candidate, True) if return_availability else candidate

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

        # Canonicalize and prove root containment before reusing this entry
        # as the next iteration's directory operand. The check must precede
        # the next os.scandir call, not run only after the walk has completed.
        try:
            resolved_candidate = candidate.resolve(strict=True)
            resolved_candidate.relative_to(workspace)
        except (OSError, RuntimeError, ValueError):
            raise SafePathError("The selected workspace entry escaped the trusted root.") from None
        if resolved_candidate == workspace:
            raise SafePathError("The workspace root itself is not a file entry.")
        current = resolved_candidate
        if final_component and return_availability:
            return current, False

    if current == workspace or not within_workspace(current):
        raise SafePathError("The path must stay inside the workspace.")
    return current

def _selected_entry(directory: int | Path, requested_name: str) -> tuple[str, str | os.PathLike[str], os.stat_result, bool]:
    """Select an entry by comparing an untrusted name to a real directory listing.

    The request string is never used as a filesystem operand. Only the name
    and path returned by the enumerated directory entry cross the open boundary.
    """
    try:
        with os.scandir(directory) as entries:
            selected = next((entry for entry in entries if entry.name == requested_name), None)
            if selected is None:
                raise FileNotFoundError("Workspace path component was not found.")
            # Windows DirEntry metadata can omit a stable file index.
            # Re-stat the OS-selected entry path so its identity is comparable
            # with fstat() on the resulting handle. POSIX descriptor scans stay
            # anchored to their pinned parent descriptor.
            metadata = (
                selected.stat(follow_symlinks=False)
                if isinstance(directory, int)
                else os.stat(selected.path, follow_symlinks=False)
            )
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            is_link_like = (
                selected.is_symlink()
                or stat.S_ISLNK(metadata.st_mode)
                or bool(getattr(metadata, "st_reparse_tag", 0))
                or bool(
                    reparse_flag
                    and getattr(metadata, "st_file_attributes", 0) & reparse_flag
                )
            )
            return selected.name, selected.path, metadata, is_link_like
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SafePathError("A workspace path component could not be inspected safely.") from exc


def _same_file_identity(expected: os.stat_result, opened: os.stat_result) -> bool:
    if os.name == "nt":
        # The Windows file index identifies the entry within its volume. The
        # drive/device field can differ between path stat and CRT fstat even
        # when both refer to the same open file handle.
        return bool(expected.st_ino and opened.st_ino and expected.st_ino == opened.st_ino)
    return (expected.st_dev, expected.st_ino) == (opened.st_dev, opened.st_ino)


def open_regular_file(path: Path) -> BinaryIO:
    """Open a regular file by walking enumerated directory entries.

    POSIX uses descriptor-relative opens with O_NOFOLLOW and verifies each
    opened descriptor against metadata from its pinned parent directory.
    Platforms without descriptor-relative opens enumerate every component and
    verify the opened handle identity before returning a stream.
    """
    target = Path(path)
    if ".." in target.parts:
        raise SafePathError("path must not contain parent-directory components")

    absolute = target if target.is_absolute() else Path.cwd() / target
    if not absolute.anchor:
        raise SafePathError("path must identify a regular file")
    components = absolute.parts[1:]
    if not components:
        raise SafePathError("path must identify a regular file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor_relative = (
        os.name == "posix"
        and bool(nofollow)
        and bool(getattr(os, "O_DIRECTORY", 0))
        and os.open in getattr(os, "supports_dir_fd", set())
        and os.scandir in getattr(os, "supports_fd", set())
    )

    if descriptor_relative:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY")
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_fd: int | None = os.open(absolute.anchor, directory_flags)
        file_fd: int | None = None
        try:
            for index, requested_name in enumerate(components):
                final_component = index == len(components) - 1
                selected_name, _selected_path, expected, is_link_like = _selected_entry(
                    parent_fd,
                    requested_name,
                )
                if is_link_like:
                    raise SafePathError("path must not contain symbolic links or junctions")

                if final_component:
                    if not stat.S_ISREG(expected.st_mode):
                        raise SafePathError("path must refer to a regular file")
                    file_fd = os.open(
                        selected_name,
                        flags | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=parent_fd,
                    )
                    opened = os.fstat(file_fd)
                    if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(expected, opened):
                        raise SafePathError("path changed while the regular file was being opened")
                    stream = os.fdopen(file_fd, "rb")
                    file_fd = None
                    return stream

                if not stat.S_ISDIR(expected.st_mode):
                    raise SafePathError("path parents must be regular directories")
                next_fd = os.open(selected_name, directory_flags, dir_fd=parent_fd)
                try:
                    opened = os.fstat(next_fd)
                    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(expected, opened):
                        raise SafePathError("path parent changed while it was being opened")
                except BaseException:
                    os.close(next_fd)
                    raise
                previous_fd = parent_fd
                parent_fd = next_fd
                os.close(previous_fd)
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if parent_fd is not None:
                os.close(parent_fd)
        raise SafePathError("path must identify a regular file")

    current: Path | int = Path(absolute.anchor)
    for index, requested_name in enumerate(components):
        final_component = index == len(components) - 1
        _selected_name, selected_path, expected, is_link_like = _selected_entry(
            current,
            requested_name,
        )
        if is_link_like:
            raise SafePathError("path must not contain symbolic links or junctions")
        selected = Path(selected_path)

        if final_component:
            if not stat.S_ISREG(expected.st_mode):
                raise SafePathError("path must refer to a regular file")
            file_fd = os.open(selected, flags | nofollow | getattr(os, "O_NONBLOCK", 0))
            try:
                opened = os.fstat(file_fd)
                if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(expected, opened):
                    raise SafePathError("path changed while the regular file was being opened")
                stream = os.fdopen(file_fd, "rb")
                file_fd = -1
                return stream
            finally:
                if file_fd >= 0:
                    os.close(file_fd)

        if not stat.S_ISDIR(expected.st_mode):
            raise SafePathError("path parents must be regular directories")
        current = selected

    raise SafePathError("path must identify a regular file")
