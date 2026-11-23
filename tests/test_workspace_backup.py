from __future__ import annotations

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from axiom.core import workspace_backup
from axiom.core.knowledge import KnowledgeBaseStore
from axiom.core.workspace_backup import (
    WorkspaceBackupError,
    create_workspace_backup,
    restore_workspace_backup,
)


def _archive(root: Path) -> tuple[io.BytesIO, dict]:
    output = io.BytesIO()
    report = create_workspace_backup(root, output)
    output.seek(0)
    return output, report


def _staging(tmp_path: Path) -> Path:
    destination = tmp_path / "staging"
    destination.mkdir(mode=0o700)
    return destination


def test_published_backup_manifest_schema_and_example_match_runtime_limits(tmp_path: Path):
    from jsonschema import Draft202012Validator, FormatChecker

    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "docs/schemas/workspace-backup-manifest-v1.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples/workspace-backup-manifest-v1.example.json").read_text(encoding="utf-8"))

    # Validate the published schema itself and both the published example and
    # a manifest emitted by the production ZIP writer with a standards-based
    # Draft 2020-12 validator (including date-time format checking).
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(example)

    workspace = tmp_path / "schema-validated-workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("AXIOM backup schema smoke test\\n", encoding="utf-8")
    archive, _report = _archive(workspace)
    with zipfile.ZipFile(archive) as bundle:
        generated_manifest = json.loads(bundle.read("manifest.json"))
    validator.validate(generated_manifest)

    assert schema["properties"]["schema_version"]["const"] == workspace_backup.BACKUP_SCHEMA_VERSION
    assert schema["properties"]["files"]["maxItems"] == workspace_backup.MAX_BACKUP_FILES
    assert example["limits"] == {
        "maximum_files": workspace_backup.MAX_BACKUP_FILES,
        "maximum_file_bytes": workspace_backup.MAX_BACKUP_FILE_BYTES,
        "maximum_total_bytes": workspace_backup.MAX_BACKUP_TOTAL_BYTES,
    }
    assert example["summary"]["total_bytes"] == sum(item["size_bytes"] for item in example["files"])


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside", "/absolute/path", "C:/Windows/boot.ini", "folder\\\\escape", "CON.txt", "a/../b"],
)
def test_workspace_backup_path_validation_rejects_cross_platform_traversal(unsafe_path: str):
    with pytest.raises(WorkspaceBackupError):
        workspace_backup._safe_relative_path(unsafe_path)


def test_workspace_backup_round_trips_regular_files_with_manifest_checksums(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "README.md").write_text("# AXIOM workspace\n", encoding="utf-8")
    metadata = workspace / ".axiom"
    metadata.mkdir()
    (metadata / "models.json").write_text("[]\n", encoding="utf-8")
    model = workspace / "models"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"local tensor bytes")
    dataset = workspace / "data"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"prompt":"private local row"}\n', encoding="utf-8")
    (workspace / ".venv").mkdir()
    (workspace / ".venv" / "ignored.txt").write_text("not included", encoding="utf-8")

    output, report = _archive(workspace)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        names = set(archive.namelist())
        assert names == {
            "manifest.json",
            "files/README.md",
            "files/.axiom/models.json",
            "files/models/model.safetensors",
            "files/data/train.jsonl",
        }
        assert str(workspace).encode() not in archive.read("manifest.json")
        assert manifest["summary"]["file_count"] == 4
        assert report["manifest_sha256"]

    staging = _staging(tmp_path)
    restored = restore_workspace_backup(output, staging)
    assert restored["integrity"] == "verified"
    assert restored["file_count"] == 4
    assert (staging / "README.md").read_text(encoding="utf-8") == "# AXIOM workspace\n"
    assert (staging / ".axiom" / "models.json").read_text(encoding="utf-8") == "[]\n"
    assert (staging / "models" / "model.safetensors").read_bytes() == b"local tensor bytes"
    assert (staging / "data" / "train.jsonl").read_text(encoding="utf-8") == '{"prompt":"private local row"}\n'
    assert (staging / ".venv").exists() is False


def test_workspace_backup_skips_symlinks_and_special_files(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside sentinel", encoding="utf-8")
    try:
        (workspace / "linked-file").symlink_to(outside / "secret.txt")
        (workspace / "linked-dir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    (workspace / "included.txt").write_text("safe", encoding="utf-8")

    output, report = _archive(workspace)
    assert report["skipped"]["symlink_files"] == 1
    assert report["skipped"]["symlink_directories"] == 1
    with zipfile.ZipFile(output) as archive:
        assert "files/included.txt" in archive.namelist()
        assert all("linked" not in name for name in archive.namelist())
        assert b"outside sentinel" not in b"".join(archive.read(name) for name in archive.namelist())


def test_workspace_backup_excludes_the_requested_output_file(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    output_path = workspace / "workspace-backup.zip"
    output_path.write_bytes(b"previous archive")
    buffer = io.BytesIO()
    report = create_workspace_backup(workspace, buffer, exclude_output=output_path)
    assert report["file_count"] == 0
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        assert archive.namelist() == ["manifest.json"]


def test_workspace_backup_fails_when_files_exceed_declared_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "large.bin").write_bytes(b"x" * 32)
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_FILE_BYTES", 16)
    with pytest.raises(WorkspaceBackupError, match="50 GiB"):
        create_workspace_backup(workspace, io.BytesIO())


def test_workspace_restore_rejects_path_traversal_and_removes_partial_staging(tmp_path: Path):
    staging = _staging(tmp_path)
    manifest = {
        "schema_version": 1,
        "backup_id": "a" * 32,
        "created_at": "2026-11-22T00:00:00+00:00",
        "workspace_label": "project",
        "files": [{
            "path": "../outside.txt",
            "size_bytes": 1,
            "sha256": "b" * 64,
            "mode": 0o600,
        }],
        "summary": {"file_count": 1, "total_bytes": 1},
        "skipped": {
            "symlink_files": 0,
            "symlink_directories": 0,
            "special_files": 0,
            "excluded_directories": 0,
        },
        "limits": {
            "maximum_files": workspace_backup.MAX_BACKUP_FILES,
            "maximum_file_bytes": workspace_backup.MAX_BACKUP_FILE_BYTES,
            "maximum_total_bytes": workspace_backup.MAX_BACKUP_TOTAL_BYTES,
        },
        "scope_note": "test",
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("files/../outside.txt", "x")
    archive.seek(0)
    outside = tmp_path / "outside.txt"
    with pytest.raises(WorkspaceBackupError):
        restore_workspace_backup(archive, staging)
    assert outside.exists() is False
    assert not staging.exists()


def test_workspace_restore_rejects_tampered_file_checksum_and_cleans_staging(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "data.txt").write_bytes(b"original")
    original, _report = _archive(workspace)

    tampered = io.BytesIO()
    with zipfile.ZipFile(original, "r") as source, zipfile.ZipFile(
        tampered, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "files/data.txt":
                data = b"tampered"
            target.writestr(info.filename, data)
    tampered.seek(0)

    staging = _staging(tmp_path)
    with pytest.raises(WorkspaceBackupError, match="size|SHA-256"):
        restore_workspace_backup(tampered, staging)
    assert not staging.exists()


def test_workspace_restore_rejects_symlink_entries(tmp_path: Path):
    staging = _staging(tmp_path)
    archive = io.BytesIO()
    manifest = {
        "schema_version": 1,
        "backup_id": "c" * 32,
        "created_at": "2026-11-22T00:00:00+00:00",
        "workspace_label": "project",
        "files": [],
        "summary": {"file_count": 0, "total_bytes": 0},
        "skipped": {
            "symlink_files": 0,
            "symlink_directories": 0,
            "special_files": 0,
            "excluded_directories": 0,
        },
        "limits": {},
        "scope_note": "test",
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        symlink = zipfile.ZipInfo("files/symlink")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(symlink, "../../outside")
    archive.seek(0)
    with pytest.raises(WorkspaceBackupError, match="non-regular"):
        restore_workspace_backup(archive, staging)
    assert not staging.exists()


def test_workspace_restore_requires_private_empty_staging_directory(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    archive, _report = _archive(workspace)

    nonempty = _staging(tmp_path)
    (nonempty / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(WorkspaceBackupError, match="empty"):
        restore_workspace_backup(archive, nonempty)
    assert (nonempty / "sentinel").read_text(encoding="utf-8") == "keep"

    archive.seek(0)
    symlinked = tmp_path / "symlinked"
    try:
        symlinked.symlink_to(nonempty, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(WorkspaceBackupError, match="real directory"):
        restore_workspace_backup(archive, symlinked)


def test_workspace_backup_restore_enforces_archive_inventory_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "one.txt").write_text("one", encoding="utf-8")
    archive, _report = _archive(workspace)
    staging = _staging(tmp_path)
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_FILES", 0)
    with pytest.raises(WorkspaceBackupError, match="entry count|file inventory"):
        restore_workspace_backup(archive, staging)
    assert not staging.exists()

def test_workspace_backup_round_trips_at_resource_limits_and_rejects_over_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "limit-boundary"
    workspace.mkdir()
    (workspace / "first.bin").write_bytes(b"a" * 16)
    (workspace / "second.bin").write_bytes(b"b" * 16)

    # Exercise the actual inventory, ZIP writer and restore checks at their
    # exact count/per-file/aggregate boundaries without allocating production
    # limits (50,000 files or tens of GiB) in the CI runner.
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_FILES", 2)
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_FILE_BYTES", 16)
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_TOTAL_BYTES", 32)

    archive, report = _archive(workspace)
    assert report["file_count"] == 2
    assert report["total_bytes"] == 32

    staging = _staging(tmp_path)
    restored = restore_workspace_backup(archive, staging)
    assert restored["file_count"] == 2
    assert (staging / "first.bin").read_bytes() == b"a" * 16
    assert (staging / "second.bin").read_bytes() == b"b" * 16

    # A third entry exceeds the file-count boundary, while one extra byte
    # exceeds the aggregate-byte boundary. Both must fail before publication.
    (workspace / "third.bin").write_bytes(b"")
    with pytest.raises(WorkspaceBackupError, match="50,000 files"):
        create_workspace_backup(workspace, io.BytesIO())

    (workspace / "third.bin").unlink()
    (workspace / "second.bin").write_bytes(b"b" * 17)
    with pytest.raises(WorkspaceBackupError, match="50 GiB"):
        create_workspace_backup(workspace, io.BytesIO())

def test_workspace_backup_preserves_ingested_knowledge_store_and_documents(tmp_path: Path):
    workspace = tmp_path / "knowledge-workspace"
    (workspace / "documents").mkdir(parents=True)
    source = workspace / "documents" / "guide.md"
    source.write_text("# Recovery guide\\nKnowledge documents are private workspace data.\\n", encoding="utf-8")

    knowledge = KnowledgeBaseStore(workspace)
    base = knowledge.create_base("Operations", "Runbooks and recovery notes")
    document = knowledge.ingest_document(base["id"], "documents/guide.md")
    expected_text = knowledge.read_document(base["id"], document["id"])

    archive, _report = _archive(workspace)
    restored = tmp_path / "restored-knowledge-workspace"
    restored.mkdir(mode=0o700)
    restore_workspace_backup(archive, restored)

    restored_knowledge = KnowledgeBaseStore(restored)
    restored_base = restored_knowledge.get_base(base["id"])
    assert restored_base is not None
    restored_documents = restored_knowledge.list_documents(base["id"])
    assert restored_documents["total"] == 1
    assert restored_documents["documents"][0]["id"] == document["id"]
    assert restored_knowledge.read_document(base["id"], document["id"]) == expected_text
    assert (restored / "documents" / "guide.md").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")



def test_workspace_backup_round_trips_the_full_production_file_count(tmp_path: Path):
    """Exercise the actual 50,000-entry ZIP64 writer and restore inventory."""
    workspace = tmp_path / "fifty-thousand-files"
    workspace.mkdir()
    for index in range(workspace_backup.MAX_BACKUP_FILES):
        (workspace / f"shard-{index:05}.bin").touch()

    archive = io.BytesIO()
    report = create_workspace_backup(workspace, archive)
    assert report["file_count"] == workspace_backup.MAX_BACKUP_FILES
    assert report["total_bytes"] == 0

    archive.seek(0)
    staging = _staging(tmp_path)
    restored = restore_workspace_backup(archive, staging)
    assert restored["file_count"] == workspace_backup.MAX_BACKUP_FILES
    assert len(list(staging.glob("shard-*.bin"))) == workspace_backup.MAX_BACKUP_FILES
    assert (staging / "shard-00000.bin").stat().st_size == 0
    assert (staging / f"shard-{workspace_backup.MAX_BACKUP_FILES - 1:05}.bin").stat().st_size == 0


def test_workspace_backup_rejects_sparse_file_above_production_per_file_limit_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Use a sparse file to exercise the 50 GiB preflight without allocating or reading 50 GiB."""
    workspace = tmp_path / "oversized-sparse-file"
    workspace.mkdir()
    sparse_file = workspace / "oversized.gguf"
    with sparse_file.open("wb") as stream:
        stream.seek(workspace_backup.MAX_BACKUP_FILE_BYTES)
        stream.write(bytes([0]))

    assert sparse_file.stat().st_size == workspace_backup.MAX_BACKUP_FILE_BYTES + 1
    monkeypatch.setattr(
        workspace_backup,
        "_hash_workspace_file",
        lambda *_args: pytest.fail("oversized files must be rejected before any hashing pass"),
    )
    with pytest.raises(WorkspaceBackupError, match="50 GiB"):
        create_workspace_backup(workspace, io.BytesIO())


def test_workspace_backup_rejects_aggregate_byte_overflow_independently_of_per_file_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "aggregate-limit"
    workspace.mkdir()
    (workspace / "one.bin").write_bytes(b"a" * 16)
    (workspace / "two.bin").write_bytes(b"b" * 16)
    (workspace / "three.bin").write_bytes(b"c")

    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_FILES", 3)
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_FILE_BYTES", 16)
    monkeypatch.setattr(workspace_backup, "MAX_BACKUP_TOTAL_BYTES", 32)

    with pytest.raises(WorkspaceBackupError, match="100 GiB"):
        create_workspace_backup(workspace, io.BytesIO())
