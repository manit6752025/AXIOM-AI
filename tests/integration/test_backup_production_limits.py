"""Opt-in production-limit workspace backup stress test for large-storage runners."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from axiom.core import workspace_backup
from axiom.core.workspace_backup import create_workspace_backup, restore_workspace_backup

pytestmark = pytest.mark.skipif(
    os.environ.get("AXIOM_RUN_BACKUP_PRODUCTION_STRESS") != "1",
    reason="Runs only on the manually dispatched large-storage validation runner.",
)


def test_workspace_backup_round_trips_full_100_gib_production_limit():
    """Stream, hash, archive and restore 100 GiB without materializing sparse zero data."""
    repository_root = Path.cwd().resolve()
    temporary_root = Path(tempfile.mkdtemp(prefix=".axiom-backup-production-stress-", dir=repository_root))
    workspace = temporary_root / "workspace"
    staging = temporary_root / "restored"
    archive_path = temporary_root / "workspace.zip"
    try:
        workspace.mkdir(mode=0o700)
        staging.mkdir(mode=0o700)
        per_file = workspace_backup.MAX_BACKUP_FILE_BYTES
        aggregate = workspace_backup.MAX_BACKUP_TOTAL_BYTES
        assert per_file * 2 == aggregate, "Expected two full-size files to exercise the 100 GiB aggregate bound."

        for name in ("model-a.gguf", "model-b.gguf"):
            path = workspace / name
            with path.open("wb") as stream:
                stream.truncate(per_file)
            assert path.stat().st_size == per_file

        with archive_path.open("w+b") as archive:
            report = create_workspace_backup(workspace, archive)
            assert report["file_count"] == 2
            assert report["total_bytes"] == aggregate

        assert archive_path.stat().st_size <= workspace_backup.MAX_BACKUP_ARCHIVE_BYTES
        with archive_path.open("rb") as archive:
            restored = restore_workspace_backup(archive, staging)

        assert restored["integrity"] == "verified"
        assert restored["file_count"] == 2
        assert restored["total_bytes"] == aggregate
        for name in ("model-a.gguf", "model-b.gguf"):
            path = staging / name
            assert path.is_file()
            assert path.stat().st_size == per_file
            if hasattr(path.stat(), "st_blocks"):
                assert path.stat().st_blocks * 512 < per_file // 64
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
