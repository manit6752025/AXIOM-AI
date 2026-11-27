"""Opt-in, near-production workspace backup stress validation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from axiom.core.workspace_backup import (
    MAX_BACKUP_FILE_BYTES,
    MAX_BACKUP_TOTAL_BYTES,
    WorkspaceBackupError,
    create_workspace_backup,
    restore_workspace_backup,
)

RUN_BACKUP_SCALE = os.environ.get("AXIOM_RUN_BACKUP_SCALE_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_BACKUP_SCALE,
    reason="Full production-size backup stress runs only in the dedicated workflow.",
)


def test_backup_restore_at_50_gib_per_file_and_100_gib_aggregate(tmp_path: Path):
    """Read, hash, archive, verify and restore exactly 100 GiB of sparse workspace data."""
    assert MAX_BACKUP_FILE_BYTES == 50 * 1024**3
    assert MAX_BACKUP_TOTAL_BYTES == 100 * 1024**3

    workspace = tmp_path / "production-limit-workspace"
    workspace.mkdir()
    paths = [workspace / "model-a.gguf", workspace / "model-b.gguf"]
    for path in paths:
        with path.open("wb") as stream:
            stream.truncate(MAX_BACKUP_FILE_BYTES)
        assert path.stat().st_size == MAX_BACKUP_FILE_BYTES

    # Use an on-disk temporary archive rather than BytesIO so the test does
    # not rely on retaining the compressed ZIP in process memory.
    with tempfile.TemporaryFile(mode="w+b", dir=tmp_path) as archive:
        report = create_workspace_backup(workspace, archive)
        assert report["file_count"] == 2
        assert report["total_bytes"] == MAX_BACKUP_TOTAL_BYTES
        archive_size = archive.tell()
        assert 0 < archive_size < 2 * 1024**3

        archive.seek(0)
        staging = tmp_path / "restored-production-limit"
        staging.mkdir(mode=0o700)
        restored = restore_workspace_backup(archive, staging)

    assert restored["integrity"] == "verified"
    assert restored["file_count"] == 2
    assert restored["total_bytes"] == MAX_BACKUP_TOTAL_BYTES
    for name in ("model-a.gguf", "model-b.gguf"):
        path = staging / name
        metadata = path.stat()
        assert metadata.st_size == MAX_BACKUP_FILE_BYTES
        with path.open("rb") as stream:
            assert stream.read(1) == b"\x00"
            stream.seek(MAX_BACKUP_FILE_BYTES // 2)
            assert stream.read(1) == b"\x00"
            stream.seek(-1, os.SEEK_END)
            assert stream.read(1) == b"\x00"
        if hasattr(metadata, "st_blocks"):
            assert metadata.st_blocks * 512 < MAX_BACKUP_FILE_BYTES // 64
