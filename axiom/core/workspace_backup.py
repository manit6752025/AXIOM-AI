"""Streaming, checksum-verified full-workspace backups and safe extraction.

Archives contain regular workspace files only. Symlinks and special files are
never restored. Restore targets must be a new private staging directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from axiom.path_safety import SafePathError, open_regular_file

BACKUP_SCHEMA_VERSION = 1
MAX_BACKUP_FILES = 50_000
MAX_BACKUP_FILE_BYTES = 50 * 1024**3
MAX_BACKUP_TOTAL_BYTES = 100 * 1024**3
MAX_BACKUP_MANIFEST_BYTES = 32 * 1024**2
MAX_BACKUP_ARCHIVE_BYTES = 105 * 1024**3
_COPY_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_BACKUP_ID = re.compile(r"^[a-f0-9]{32}$")
_EXCLUDED_DIRECTORIES = frozenset({
    ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".cache", ".next", ".turbo",
})


class WorkspaceBackupError(ValueError):
    """A workspace backup or restore failed strict validation."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Backup manifest contains duplicate JSON object keys.")
        result[key] = value
    return result


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise WorkspaceBackupError("Backup contains an invalid relative path.")
    if "\\" in value or "\x00" in value or ":" in value or value.startswith("/"):
        raise WorkspaceBackupError("Backup contains an unsafe relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspaceBackupError("Backup contains an unsafe relative path.")
    if any(
        len(part) > 255
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].upper() in {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
        for part in path.parts
    ):
        raise WorkspaceBackupError("Backup contains a path that is not portable across supported filesystems.")
    normalized = path.as_posix()
    if normalized != value:
        raise WorkspaceBackupError("Backup contains a non-canonical relative path.")
    return normalized


def _workspace_root(workspace: str | Path) -> Path:
    candidate = Path(workspace).expanduser()
    if candidate.is_symlink():
        raise WorkspaceBackupError("Workspace root must not be a symbolic link.")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkspaceBackupError("Workspace root could not be resolved safely.") from None
    if not root.is_dir():
        raise WorkspaceBackupError("Workspace root must be a directory.")
    return root


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _hash_workspace_file(path: Path, expected: os.stat_result) -> str:
    digest = hashlib.sha256()
    try:
        with open_regular_file(path) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or _file_identity(before) != _file_identity(expected):
                raise WorkspaceBackupError("A workspace file changed during backup scanning.")
            total = 0
            while True:
                block = stream.read(_COPY_CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                if total > MAX_BACKUP_FILE_BYTES:
                    raise WorkspaceBackupError("A workspace file exceeds the 50 GiB backup limit.")
                digest.update(block)
            after = os.fstat(stream.fileno())
            current = path.lstat()
    except WorkspaceBackupError:
        raise
    except (OSError, ValueError, SafePathError):
        raise WorkspaceBackupError("A workspace file could not be opened safely for backup.") from None
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(current)
        or total != expected.st_size
    ):
        raise WorkspaceBackupError("A workspace file changed while its checksum was calculated.")
    return digest.hexdigest()


def _inventory_workspace(root: Path, exclude_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    entries: list[dict[str, Any]] = []
    skipped = {
        "symlink_files": 0,
        "symlink_directories": 0,
        "special_files": 0,
        "excluded_directories": 0,
    }
    total_bytes = 0
    folded_paths: set[str] = set()
    excluded_outputs = {path.resolve(strict=False) for path in exclude_paths}

    def walk_error(_error: OSError) -> None:
        raise WorkspaceBackupError("A workspace directory could not be scanned safely.") from None

    try:
        for current, directories, filenames in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
            current_path = Path(current)
            retained_directories: list[str] = []
            for name in directories:
                child = current_path / name
                if name in _EXCLUDED_DIRECTORIES:
                    skipped["excluded_directories"] += 1
                    continue
                try:
                    if child.is_symlink():
                        skipped["symlink_directories"] += 1
                        continue
                    info = child.stat(follow_symlinks=False)
                except OSError:
                    raise WorkspaceBackupError("A workspace directory changed during backup scanning.") from None
                if not stat.S_ISDIR(info.st_mode):
                    skipped["special_files"] += 1
                    continue
                retained_directories.append(name)
            directories[:] = sorted(retained_directories, key=lambda item: (item.casefold(), item))

            for name in filenames:
                path = current_path / name
                try:
                    if path.resolve(strict=False) in excluded_outputs:
                        continue
                except (OSError, RuntimeError):
                    raise WorkspaceBackupError("A backup exclusion path could not be checked safely.") from None
                try:
                    info = path.lstat()
                except OSError:
                    raise WorkspaceBackupError("A workspace file changed during backup scanning.") from None
                if stat.S_ISLNK(info.st_mode):
                    skipped["symlink_files"] += 1
                    continue
                if not stat.S_ISREG(info.st_mode):
                    skipped["special_files"] += 1
                    continue
                relative = _safe_relative_path(path.relative_to(root).as_posix())
                folded = relative.casefold()
                if folded in folded_paths:
                    raise WorkspaceBackupError("Workspace contains paths that collide on a case-insensitive filesystem.")
                folded_paths.add(folded)
                if info.st_size < 0 or info.st_size > MAX_BACKUP_FILE_BYTES:
                    raise WorkspaceBackupError("A workspace file exceeds the 50 GiB backup limit.")
                total_bytes += info.st_size
                if total_bytes > MAX_BACKUP_TOTAL_BYTES:
                    raise WorkspaceBackupError("Workspace data exceeds the 100 GiB backup limit.")
                if len(entries) >= MAX_BACKUP_FILES:
                    raise WorkspaceBackupError("Workspace contains more than 50,000 files.")
                digest = _hash_workspace_file(path, info)
                mode = (stat.S_IMODE(info.st_mode) & 0o700) | 0o600
                entries.append({
                    "path": relative,
                    "size_bytes": info.st_size,
                    "sha256": digest,
                    "mode": mode,
                })
    except WorkspaceBackupError:
        raise
    except OSError:
        raise WorkspaceBackupError("Workspace inventory could not be completed safely.") from None

    entries.sort(key=lambda item: (item["path"].casefold(), item["path"]))
    return entries, skipped, total_bytes


def create_workspace_backup(
    workspace: str | Path,
    output: BinaryIO,
    *,
    exclude_output: str | Path | None = None,
    exclude_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Write a ZIP64 backup to a caller-owned seekable binary stream."""
    root = _workspace_root(workspace)
    exclusions = [Path(item).expanduser() for item in (exclude_paths or [])]
    if exclude_output is not None:
        exclusions.append(Path(exclude_output).expanduser())
    try:
        output_start = output.tell()
    except (OSError, AttributeError):
        raise WorkspaceBackupError("Backup output stream must be seekable.") from None
    entries, skipped, total_bytes = _inventory_workspace(root, exclusions)
    manifest = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "backup_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "workspace_label": root.name[:64],
        "files": entries,
        "summary": {
            "file_count": len(entries),
            "total_bytes": total_bytes,
        },
        "skipped": skipped,
        "limits": {
            "maximum_files": MAX_BACKUP_FILES,
            "maximum_file_bytes": MAX_BACKUP_FILE_BYTES,
            "maximum_total_bytes": MAX_BACKUP_TOTAL_BYTES,
        },
        "scope_note": (
            "Full regular-file workspace backup, including project files, .axiom metadata, datasets, and model weights. "
            "Virtual environments, dependency caches, symlinks, and special filesystem entries are not restored. "
            "The archive may contain private data or credentials from the workspace; store and share it accordingly."
        ),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(manifest_bytes) > MAX_BACKUP_MANIFEST_BYTES:
        raise WorkspaceBackupError("Backup manifest exceeds the 32 MiB limit.")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    try:
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            manifest_info = zipfile.ZipInfo("manifest.json")
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.external_attr = (stat.S_IFREG | 0o600) << 16
            with archive.open(manifest_info, "w", force_zip64=True) as target:
                target.write(manifest_bytes)

            for entry in entries:
                source = root.joinpath(*PurePosixPath(entry["path"]).parts)
                try:
                    with open_regular_file(source) as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_size != entry["size_bytes"]:
                            raise WorkspaceBackupError("A workspace file changed between backup verification passes.")
                        info = zipfile.ZipInfo(f"files/{entry['path']}")
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = (stat.S_IFREG | entry["mode"]) << 16
                        digest = hashlib.sha256()
                        copied = 0
                        with archive.open(info, "w", force_zip64=True) as target:
                            while True:
                                block = stream.read(_COPY_CHUNK_BYTES)
                                if not block:
                                    break
                                copied += len(block)
                                if copied > entry["size_bytes"]:
                                    raise WorkspaceBackupError("A workspace file grew during backup writing.")
                                digest.update(block)
                                target.write(block)
                        after = os.fstat(stream.fileno())
                        current = source.lstat()
                except WorkspaceBackupError:
                    raise
                except (OSError, ValueError, SafePathError, zipfile.BadZipFile):
                    raise WorkspaceBackupError("A workspace file could not be archived safely.") from None
                if (
                    _file_identity(before) != _file_identity(after)
                    or _file_identity(before) != _file_identity(current)
                    or copied != entry["size_bytes"]
                    or digest.hexdigest() != entry["sha256"]
                ):
                    raise WorkspaceBackupError("A workspace file changed while the backup archive was written.")
    except WorkspaceBackupError:
        raise
    except (OSError, ValueError, zipfile.LargeZipFile, zipfile.BadZipFile):
        raise WorkspaceBackupError("Workspace backup archive could not be created safely.") from None

    archive_bytes = output.tell() - output_start
    if archive_bytes <= 0 or archive_bytes > MAX_BACKUP_ARCHIVE_BYTES:
        raise WorkspaceBackupError("Workspace backup archive exceeds the 105 GiB compressed-size limit.")
    return {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "backup_id": manifest["backup_id"],
        "workspace_label": manifest["workspace_label"],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "archive_bytes": archive_bytes,
        "manifest_sha256": manifest_sha256,
        "skipped": skipped,
        "limits": manifest["limits"],
        "privacy": {
            "absolute_workspace_path_included": False,
            "environment_variables_included": False,
            "symlinks_followed": False,
        },
        "scope_note": manifest["scope_note"],
    }


def _validate_manifest(manifest: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "backup_id", "created_at", "workspace_label", "files",
        "summary", "skipped", "limits", "scope_note",
    }:
        raise WorkspaceBackupError("Backup manifest has an invalid schema.")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != BACKUP_SCHEMA_VERSION:
        raise WorkspaceBackupError("Unsupported workspace backup schema version.")
    if not isinstance(manifest.get("backup_id"), str) or not _BACKUP_ID.fullmatch(manifest["backup_id"]):
        raise WorkspaceBackupError("Backup identifier is invalid.")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or len(created_at) > 64:
        raise WorkspaceBackupError("Backup creation timestamp is invalid.")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        raise WorkspaceBackupError("Backup creation timestamp is invalid.") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise WorkspaceBackupError("Backup creation timestamp must include a timezone.")
    workspace_label = manifest.get("workspace_label")
    if (
        not isinstance(workspace_label, str)
        or not 1 <= len(workspace_label) <= 64
        or any(not character.isprintable() for character in workspace_label)
    ):
        raise WorkspaceBackupError("Backup workspace label is invalid.")
    if not isinstance(manifest.get("scope_note"), str) or len(manifest["scope_note"]) > 2048:
        raise WorkspaceBackupError("Backup scope note is invalid.")
    limits = manifest.get("limits")
    if (
        not isinstance(limits, dict)
        or set(limits) != {"maximum_files", "maximum_file_bytes", "maximum_total_bytes"}
        or type(limits.get("maximum_files")) is not int
        or limits.get("maximum_files") != MAX_BACKUP_FILES
        or type(limits.get("maximum_file_bytes")) is not int
        or limits.get("maximum_file_bytes") != MAX_BACKUP_FILE_BYTES
        or type(limits.get("maximum_total_bytes")) is not int
        or limits.get("maximum_total_bytes") != MAX_BACKUP_TOTAL_BYTES
    ):
        raise WorkspaceBackupError("Backup manifest limits do not match the supported format version.")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > MAX_BACKUP_FILES:
        raise WorkspaceBackupError("Backup file inventory exceeds its limit.")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    folded_paths: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256", "mode"}:
            raise WorkspaceBackupError("Backup manifest contains an invalid file record.")
        relative = _safe_relative_path(item.get("path"))
        size = item.get("size_bytes")
        digest = item.get("sha256")
        mode = item.get("mode")
        if (
            type(size) is not int or size < 0 or size > MAX_BACKUP_FILE_BYTES
            or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            or type(mode) is not int or mode < 0 or mode > 0o777
        ):
            raise WorkspaceBackupError("Backup manifest contains invalid file metadata.")
        folded = relative.casefold()
        if relative in seen_paths or folded in folded_paths:
            raise WorkspaceBackupError("Backup manifest contains duplicate or colliding paths.")
        seen_paths.add(relative)
        folded_paths.add(folded)
        total_bytes += size
        if total_bytes > MAX_BACKUP_TOTAL_BYTES:
            raise WorkspaceBackupError("Backup uncompressed size exceeds the 100 GiB limit.")
        normalized.append({"path": relative, "size_bytes": size, "sha256": digest, "mode": mode})
    summary = manifest.get("summary")
    if (
        not isinstance(summary, dict)
        or set(summary) != {"file_count", "total_bytes"}
        or type(summary.get("file_count")) is not int
        or summary["file_count"] != len(normalized)
        or type(summary.get("total_bytes")) is not int
        or summary["total_bytes"] != total_bytes
    ):
        raise WorkspaceBackupError("Backup manifest summary does not match its file inventory.")
    skipped = manifest.get("skipped")
    expected_skipped = {"symlink_files", "symlink_directories", "special_files", "excluded_directories"}
    if (
        not isinstance(skipped, dict)
        or set(skipped) != expected_skipped
        or any(type(skipped.get(key)) is not int or skipped[key] < 0 for key in expected_skipped)
    ):
        raise WorkspaceBackupError("Backup skipped-entry summary is invalid.")
    return normalized, {"file_count": len(normalized), "total_bytes": total_bytes, "skipped": skipped}


def restore_workspace_backup(archive_source: BinaryIO, staging_root: str | Path) -> dict[str, Any]:
    """Verify and extract a backup into an empty, caller-owned private staging directory."""
    root = Path(staging_root)
    if root.is_symlink() or not root.is_dir():
        raise WorkspaceBackupError("Restore staging destination must be a real directory.")
    try:
        if any(root.iterdir()):
            raise WorkspaceBackupError("Restore staging destination must be empty.")
        root_info = root.stat()
        if os.name == "posix" and root_info.st_mode & 0o077:
            raise WorkspaceBackupError("Restore staging destination must be private to the current user.")
    except WorkspaceBackupError:
        raise
    except OSError:
        raise WorkspaceBackupError("Restore staging destination could not be inspected safely.") from None

    try:
        current_position = archive_source.tell()
        archive_source.seek(0, os.SEEK_END)
        archive_size = archive_source.tell()
        archive_source.seek(current_position)
    except (OSError, AttributeError):
        raise WorkspaceBackupError("Backup archive source must be seekable.") from None
    if archive_size <= 0 or archive_size > MAX_BACKUP_ARCHIVE_BYTES:
        raise WorkspaceBackupError("Backup archive exceeds the 105 GiB upload limit.")

    try:
        archive = zipfile.ZipFile(archive_source, mode="r")
    except (OSError, zipfile.BadZipFile):
        raise WorkspaceBackupError("Backup file is not a valid ZIP archive.") from None

    try:
        with archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_BACKUP_FILES + 1:
                raise WorkspaceBackupError("Backup archive entry count is invalid.")
            by_name: dict[str, zipfile.ZipInfo] = {}
            folded_names: set[str] = set()
            for info in infos:
                name = info.filename
                if (
                    not isinstance(name, str)
                    or not name
                    or name.startswith("/")
                    or "\\" in name
                    or "\x00" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                ):
                    raise WorkspaceBackupError("Backup archive contains an unsafe ZIP entry.")
                folded = name.casefold()
                if name in by_name or folded in folded_names:
                    raise WorkspaceBackupError("Backup archive contains duplicate or colliding ZIP entries.")
                file_type = stat.S_IFMT(info.external_attr >> 16)
                if file_type not in {0, stat.S_IFREG}:
                    raise WorkspaceBackupError("Backup archive contains a non-regular filesystem entry.")
                if info.file_size < 0 or info.file_size > MAX_BACKUP_MANIFEST_BYTES + MAX_BACKUP_TOTAL_BYTES:
                    raise WorkspaceBackupError("Backup archive contains an oversized entry.")
                by_name[name] = info
                folded_names.add(folded)

            manifest_info = by_name.get("manifest.json")
            if manifest_info is None or manifest_info.file_size > MAX_BACKUP_MANIFEST_BYTES:
                raise WorkspaceBackupError("Backup archive has no bounded manifest.json.")
            with archive.open(manifest_info, "r") as stream:
                manifest_bytes = stream.read(MAX_BACKUP_MANIFEST_BYTES + 1)
            if len(manifest_bytes) > MAX_BACKUP_MANIFEST_BYTES:
                raise WorkspaceBackupError("Backup manifest exceeds the 32 MiB limit.")
            try:
                manifest = json.loads(
                    manifest_bytes.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_keys,
                )
            except (UnicodeError, json.JSONDecodeError, ValueError):
                raise WorkspaceBackupError("Backup manifest is not valid duplicate-free UTF-8 JSON.") from None
            entries, summary = _validate_manifest(manifest)
            expected_names = {"manifest.json", *(f"files/{item['path']}" for item in entries)}
            if set(by_name) != expected_names:
                raise WorkspaceBackupError("Backup ZIP entries do not match the manifest inventory.")
            for entry in entries:
                info = by_name[f"files/{entry['path']}"]
                if info.file_size != entry["size_bytes"]:
                    raise WorkspaceBackupError("Backup file size does not match the manifest.")

            for entry in entries:
                relative = _safe_relative_path(entry["path"])
                destination = root.joinpath(*PurePosixPath(relative).parts)
                parent = destination.parent
                try:
                    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    current = root
                    for component in PurePosixPath(relative).parts[:-1]:
                        current = current / component
                        if current.is_symlink() or not current.is_dir():
                            raise WorkspaceBackupError("Restore path contains an unsafe directory component.")
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
                    descriptor = os.open(destination, flags, 0o600)
                except WorkspaceBackupError:
                    raise
                except OSError:
                    raise WorkspaceBackupError("A restore file could not be created safely.") from None
                digest = hashlib.sha256()
                written = 0
                try:
                    with archive.open(by_name[f"files/{relative}"], "r") as source:
                        with os.fdopen(descriptor, "wb") as target:
                            descriptor = -1
                            while True:
                                block = source.read(_COPY_CHUNK_BYTES)
                                if not block:
                                    break
                                written += len(block)
                                if written > entry["size_bytes"]:
                                    raise WorkspaceBackupError("A restored file exceeded its declared size.")
                                digest.update(block)
                                # Preserve long all-zero extents as filesystem holes
                                # instead of allocating every byte during restore.
                                # The content is still fully read, hashed, CRC-checked
                                # by zipfile, and included in the declared byte count.
                                if block.count(0) != len(block):
                                    target.write(block)
                                else:
                                    target.seek(len(block), os.SEEK_CUR)
                            target.truncate(entry["size_bytes"])
                            target.flush()
                            os.fsync(target.fileno())
                    if written != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                        raise WorkspaceBackupError("A restored file failed its SHA-256 or size check.")
                    safe_mode = (entry["mode"] & 0o755) | 0o600
                    os.chmod(destination, safe_mode)
                except WorkspaceBackupError:
                    raise
                except (OSError, RuntimeError, zipfile.BadZipFile, EOFError):
                    raise WorkspaceBackupError("A backup entry could not be extracted and verified.") from None
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)

            return {
                "schema_version": BACKUP_SCHEMA_VERSION,
                "backup_id": manifest["backup_id"],
                "workspace_label": manifest["workspace_label"],
                "file_count": summary["file_count"],
                "total_bytes": summary["total_bytes"],
                "skipped": summary["skipped"],
                "integrity": "verified",
                "scope_note": "The archive manifest, paths, sizes, ZIP CRCs, and SHA-256 checksums were verified before the new workspace was registered.",
            }
    except WorkspaceBackupError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, EOFError, ValueError):
        shutil.rmtree(root, ignore_errors=True)
        raise WorkspaceBackupError("Backup archive was rejected; the restore staging directory was removed.") from None
