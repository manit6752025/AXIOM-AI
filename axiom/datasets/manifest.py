"""Bounded, value-free dataset provenance manifests."""
from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from typing import Any, BinaryIO, Callable

DEFAULT_MANIFEST_MAX_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_MAX_BYTES = 1024 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024


def create_dataset_manifest(
    stream: BinaryIO,
    *,
    file_name: str,
    relative_path: str,
    file_size_bytes: int,
    max_bytes: int = DEFAULT_MANIFEST_MAX_BYTES,
    parent_manifest_sha256: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Hash a bounded prefix, claiming a full digest only for a stable full read."""
    if type(file_size_bytes) is not int or file_size_bytes < 0:
        raise ValueError("File size must be a non-negative integer.")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_MANIFEST_MAX_BYTES:
        raise ValueError("Manifest scan limit must be between 1 byte and 1 GiB.")
    if parent_manifest_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", parent_manifest_sha256):
        raise ValueError("Parent manifest SHA-256 must contain exactly 64 hexadecimal characters.")

    try:
        initial_stat = os.fstat(stream.fileno())
    except (AttributeError, OSError):
        initial_stat = None

    expected_bytes = min(file_size_bytes, max_bytes)
    digest = hashlib.sha256()
    bytes_hashed = 0
    while bytes_hashed < expected_bytes:
        chunk = stream.read(min(HASH_CHUNK_BYTES, expected_bytes - bytes_hashed))
        if not chunk:
            break
        digest.update(chunk)
        bytes_hashed += len(chunk)
        if on_progress is not None:
            on_progress(bytes_hashed, expected_bytes)

    if on_progress is not None and bytes_hashed == 0:
        on_progress(0, expected_bytes)

    try:
        final_stat = os.fstat(stream.fileno())
    except (AttributeError, OSError):
        final_stat = None
    stable = (
        initial_stat is None or final_stat is None
        or (initial_stat.st_size == final_stat.st_size and initial_stat.st_mtime_ns == final_stat.st_mtime_ns)
    )
    full_read = bytes_hashed == file_size_bytes
    status = "complete" if full_read and stable else "changed_during_read" if not stable else "partial"
    fingerprint = digest.hexdigest()

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "path": relative_path,
        "lineage": {
            "relation": "derived_from" if parent_manifest_sha256 else "source",
            "parent_manifest_sha256": parent_manifest_sha256.lower() if parent_manifest_sha256 else None,
        },
        "file_name": file_name,
        "format": "jsonl",
        "file_size_bytes": file_size_bytes,
        "bytes_hashed": bytes_hashed,
        "full_content_hashed": full_read and stable,
        "sha256": fingerprint if full_read and stable else None,
        "partial_sha256": fingerprint if not full_read and stable else None,
        "status": status,
        "modified_at": (
            datetime.fromtimestamp(initial_stat.st_mtime, UTC).isoformat()
            if initial_stat is not None else None
        ),
        "limits": {"max_bytes": max_bytes},
    }
