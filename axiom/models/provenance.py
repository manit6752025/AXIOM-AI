"""Bounded local model provenance and security evidence inspection.

The report distinguishes hashes and filesystem evidence from claims entered in
AXIOM's model registry. It never contacts a remote source or executes model code.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from axiom.path_safety import open_regular_file

MAX_PROVENANCE_DIRECTORIES = 2_000
MAX_PROVENANCE_FILES = 20_000
MAX_PROVENANCE_WEIGHT_FILES = 512
MAX_PROVENANCE_HASH_BYTES = 16 * 1024**3
MAX_LICENSE_EVIDENCE_BYTES = 1 * 1024**2
MAX_LICENSE_EVIDENCE_FILES = 64
MAX_TOTAL_LICENSE_EVIDENCE_BYTES = 16 * 1024**2
_READ_CHUNK_BYTES = 1024 * 1024
_WEIGHT_FORMATS = {
    ".safetensors": "safetensors",
    ".gguf": "gguf",
    ".bin": "pytorch_pickle_or_binary",
    ".pt": "pytorch_pickle_or_binary",
    ".pth": "pytorch_pickle_or_binary",
}
_LICENSE_NAMES = frozenset({
    "license", "license.md", "license.txt", "copying", "copying.md",
    "copying.txt", "notice", "notice.md", "notice.txt",
})


class ModelProvenanceError(ValueError):
    """A model provenance scan failed its local filesystem safety boundary."""


def _hash_file(path: Path, *, expected_size: int, max_bytes: int) -> str | None:
    try:
        path_info = path.lstat()
    except OSError:
        raise ModelProvenanceError("A model file changed during provenance inspection.") from None
    if not stat.S_ISREG(path_info.st_mode) or path_info.st_size != expected_size:
        raise ModelProvenanceError("A model file changed during provenance inspection.")
    if expected_size > max_bytes:
        return None
    digest = hashlib.sha256()
    try:
        with open_regular_file(path) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise ModelProvenanceError("A model file changed during provenance inspection.")
            total = 0
            while True:
                block = stream.read(_READ_CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    return None
                digest.update(block)
            after = os.fstat(stream.fileno())
            current = path.lstat()
    except ModelProvenanceError:
        raise
    except (OSError, ValueError):
        raise ModelProvenanceError("A model file could not be opened safely for hashing.") from None
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_path = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if identity_before != identity_after or identity_before != identity_path or total != expected_size:
        raise ModelProvenanceError("A model file changed during provenance inspection.")
    return digest.hexdigest()


def inspect_model_provenance(
    path: str | Path,
    *,
    registry_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return bounded SHA-256, weight-format, and license-file evidence."""
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ModelProvenanceError("Model directory must not be a symbolic link.")
    try:
        root = requested.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ModelProvenanceError("Model directory is unavailable.") from None
    if not root.is_dir():
        raise ModelProvenanceError("Model provenance path must be a directory.")

    files: list[Path] = []
    symlink_files_skipped = 0
    symlink_directories_skipped = 0
    directory_count = 0

    def walk_error(_error: OSError) -> None:
        raise ModelProvenanceError("A model directory could not be enumerated safely.") from None

    try:
        for current, subdirectories, filenames in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
            directory_count += 1
            if directory_count > MAX_PROVENANCE_DIRECTORIES:
                raise ModelProvenanceError("Model contains too many directories for a bounded provenance scan.")
            current_path = Path(current)
            retained_directories = []
            for name in subdirectories:
                child = current_path / name
                if child.is_symlink():
                    symlink_directories_skipped += 1
                else:
                    retained_directories.append(name)
            subdirectories[:] = sorted(retained_directories, key=str.casefold)
            for name in filenames:
                child = current_path / name
                if child.is_symlink():
                    symlink_files_skipped += 1
                    continue
                try:
                    info = child.stat(follow_symlinks=False)
                except OSError:
                    raise ModelProvenanceError("A model file could not be inspected safely.") from None
                if stat.S_ISREG(info.st_mode):
                    files.append(child)
                    if len(files) > MAX_PROVENANCE_FILES:
                        raise ModelProvenanceError("Model contains too many files for a bounded provenance scan.")
    except ModelProvenanceError:
        raise
    except OSError:
        raise ModelProvenanceError("Model directory could not be scanned safely.") from None

    weight_files: list[tuple[str, Path, int, str]] = []
    license_files: list[tuple[str, Path, int]] = []
    total_weight_bytes = 0
    format_counts: dict[str, int] = {}
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        suffix = file_path.suffix.casefold()
        try:
            file_info = file_path.stat(follow_symlinks=False)
        except OSError:
            raise ModelProvenanceError("A model file changed during provenance inspection.") from None
        if not stat.S_ISREG(file_info.st_mode):
            raise ModelProvenanceError("A model file changed during provenance inspection.")
        size = file_info.st_size
        if suffix in _WEIGHT_FORMATS:
            fmt = _WEIGHT_FORMATS[suffix]
            weight_files.append((relative, file_path, size, fmt))
            total_weight_bytes += size
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if file_path.name.casefold() in _LICENSE_NAMES:
            license_files.append((relative, file_path, size))

    weight_files.sort(key=lambda item: (item[0].casefold(), item[0]))
    license_files.sort(key=lambda item: (item[0].casefold(), item[0]))
    hash_allowed = (
        bool(weight_files)
        and len(weight_files) <= MAX_PROVENANCE_WEIGHT_FILES
        and total_weight_bytes <= MAX_PROVENANCE_HASH_BYTES
    )
    hashed_files: list[dict[str, Any]] = []
    checkpoint_digest = None
    if hash_allowed:
        tree_digest = hashlib.sha256()
        for relative, file_path, size, fmt in weight_files:
            file_digest = _hash_file(file_path, expected_size=size, max_bytes=MAX_PROVENANCE_HASH_BYTES)
            if file_digest is None:
                hash_allowed = False
                hashed_files = []
                break
            hashed_files.append({
                "path": relative,
                "format": fmt,
                "size_bytes": size,
                "sha256": file_digest,
            })
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(b"\0")
            tree_digest.update(str(size).encode("ascii"))
            tree_digest.update(b"\0")
            tree_digest.update(bytes.fromhex(file_digest))
        if hash_allowed:
            checkpoint_digest = tree_digest.hexdigest()

    total_license_bytes = sum(item[2] for item in license_files)
    license_scan_allowed = (
        len(license_files) <= MAX_LICENSE_EVIDENCE_FILES
        and total_license_bytes <= MAX_TOTAL_LICENSE_EVIDENCE_BYTES
    )
    local_license_files = []
    if license_scan_allowed:
        for relative, file_path, size in license_files:
            digest = _hash_file(file_path, expected_size=size, max_bytes=MAX_LICENSE_EVIDENCE_BYTES)
            local_license_files.append({
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
                "hash_status": "complete" if digest is not None else "file_size_limit",
            })

    if not weight_files:
        hash_status = "no_weight_files"
    elif hash_allowed:
        hash_status = "complete"
    else:
        hash_status = "skipped_size_or_file_count_limit"

    formats = sorted(format_counts)
    registry = registry_metadata if isinstance(registry_metadata, dict) else {}
    registry_text = {
        key: value[:2048] if isinstance(value, str) else None
        for key, value in registry.items()
        if key in {"source", "source_url", "source_revision", "source_attribution", "license",
                   "license_review", "license_review_notes"}
    }
    source_claims_present = any(registry_text.get(key) for key in (
        "source", "source_url", "source_revision", "source_attribution",
    ))
    return {
        "schema_version": 1,
        "model_directory": str(root),
        "registry_metadata": {
            "status": "supplied" if registry_text else "not_linked",
            "evidence_origin": "user_supplied_workspace_registry" if registry_text else None,
            **registry_text,
        },
        "source_evidence": {
            "status": "registry_claim_not_verified" if source_claims_present else "not_independently_verified",
            "declared_source": registry_text.get("source"),
            "source_url": registry_text.get("source_url"),
            "source_revision": registry_text.get("source_revision"),
            "source_attribution": registry_text.get("source_attribution"),
            "network_used": False,
            "note": "Registry source details are user-supplied claims. A local directory and URL do not prove publisher identity, repository origin, or that the files came from the declared revision.",
        },
        "format_evidence": {
            "detected_formats": formats,
            "weight_file_count": len(weight_files),
            "weight_size_bytes": total_weight_bytes,
            "format_file_counts": {key: format_counts[key] for key in sorted(format_counts)},
            "pickle_capable_weight_files_present": any(item[3] == "pytorch_pickle_or_binary" for item in weight_files),
            "files": [
                {"path": relative, "format": fmt, "size_bytes": size}
                for relative, _file, size, fmt in weight_files[:MAX_PROVENANCE_WEIGHT_FILES]
            ],
        },
        "hash_evidence": {
            "algorithm": "SHA-256",
            "status": hash_status,
            "checkpoint_sha256": checkpoint_digest,
            "hashed_file_count": len(hashed_files),
            "files": hashed_files,
            "maximum_total_weight_bytes": MAX_PROVENANCE_HASH_BYTES,
            "maximum_weight_files": MAX_PROVENANCE_WEIGHT_FILES,
        },
        "license_evidence": {
            "status": (
                "skipped_count_or_total_size_limit"
                if license_files and not license_scan_allowed
                else "local_files_found" if local_license_files else "no_local_license_file_found"
            ),
            "candidate_file_count": len(license_files),
            "candidate_total_bytes": total_license_bytes,
            "maximum_file_bytes": MAX_LICENSE_EVIDENCE_BYTES,
            "maximum_files": MAX_LICENSE_EVIDENCE_FILES,
            "maximum_total_bytes": MAX_TOTAL_LICENSE_EVIDENCE_BYTES,
            "human_review_required": True,
            "declared_license": registry_text.get("license"),
            "registry_review_state": registry_text.get("license_review"),
            "registry_review_notes": registry_text.get("license_review_notes"),
            "files": local_license_files,
            "note": "Registry license and review fields are user-supplied metadata. Presence and hashes of local license/notice files are evidence only; they do not establish license identity, grant rights, or replace human legal review.",
        },
        "security_evidence": {
            "symlink_files_skipped": symlink_files_skipped,
            "symlink_directories_skipped": symlink_directories_skipped,
            "remote_code_executed": False,
            "network_used": False,
            "model_weights_loaded": False,
        },
        "scope_note": (
            "This is a read-only local metadata scan. SHA-256 identifies the scanned weight-file bytes and relative paths, "
            "but is not publisher authentication or a malware verdict. Pickle-capable PyTorch weight formats are flagged; "
            "do not load untrusted pickle checkpoints."
        ),
    }
