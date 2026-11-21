from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from axiom.core.workspaces import WorkspaceRegistry
from axiom.cli.main import app


def test_workspace_backup_cli_create_and_restore_registers_new_workspace(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / "catalog"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / "managed"))
    (workspace / "axiom.yaml").write_text("project:\n  name: test\n", encoding="utf-8")
    (workspace / "README.md").write_text("workspace bytes", encoding="utf-8")
    (workspace / "data").mkdir()
    (workspace / "data" / "train.jsonl").write_text('{"text":"private"}\n', encoding="utf-8")
    archive = tmp_path / "workspace.zip"
    runner = CliRunner()

    created = runner.invoke(app, [
        "--no-banner",
        "workspace", "backup", "create",
        "--workspace", str(workspace),
        "--output", str(archive),
        "--json",
    ])
    assert created.exit_code == 0, created.output
    report = json.loads(created.stdout)
    assert report["status"] == "created"
    assert report["file_count"] == 3
    assert archive.is_file()
    assert report["archive_sha256"]
    assert report["manifest_sha256"]

    restored = runner.invoke(app, [
        "--no-banner",
        "workspace", "backup", "restore",
        str(archive),
        "--name", "CLI restored workspace",
        "--json",
    ])
    assert restored.exit_code == 0, restored.output
    payload = json.loads(restored.stdout)
    assert payload["status"] == "restored"
    restored_id = payload["workspace"]["id"]
    restored_root = WorkspaceRegistry().resolve(restored_id).root
    assert (restored_root / "README.md").read_text(encoding="utf-8") == "workspace bytes"
    assert (restored_root / "data" / "train.jsonl").read_text(encoding="utf-8") == '{"text":"private"}\n'
    assert (workspace / "README.md").read_text(encoding="utf-8") == "workspace bytes"


def test_workspace_backup_cli_never_overwrites_existing_archive(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    archive = tmp_path / "already-exists.zip"
    archive.write_bytes(b"do not overwrite")
    runner = CliRunner()
    result = runner.invoke(app, [
        "--no-banner",
        "workspace", "backup", "create",
        "--workspace", str(workspace),
        "--output", str(archive),
    ])
    assert result.exit_code != 0
    assert archive.read_bytes() == b"do not overwrite"


def test_workspace_backup_cli_rejects_symlinked_archive_input(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    real_archive = tmp_path / "real.zip"
    real_archive.write_bytes(b"not a backup")
    linked = tmp_path / "linked.zip"
    try:
        linked.symlink_to(real_archive)
    except OSError:
        import pytest
        pytest.skip("symlink creation is unavailable")
    runner = CliRunner()
    result = runner.invoke(app, [
        "--no-banner", "workspace", "backup", "restore",
        str(linked), "--name", "Rejected",
    ])
    assert result.exit_code != 0
    assert real_archive.read_bytes() == b"not a backup"
