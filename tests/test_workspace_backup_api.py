from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from axiom.core.workspaces import WorkspaceRegistry


def _client(tmp_path: Path, monkeypatch):
    default = tmp_path / "project"
    default.mkdir()
    monkeypatch.chdir(default)
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / "catalog"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / "managed"))
    from axiom.api.server import create_app
    return TestClient(create_app()), default


def test_workspace_backup_api_downloads_and_restores_as_separate_workspace(tmp_path: Path, monkeypatch):
    client, default = _client(tmp_path, monkeypatch)
    (default / "README.md").write_text("original workspace", encoding="utf-8")
    (default / ".axiom").mkdir()
    (default / ".axiom" / "models.json").write_text("[]\n", encoding="utf-8")
    (default / "data").mkdir()
    (default / "data" / "private.jsonl").write_text('{"row":"private"}\n', encoding="utf-8")

    catalog = client.get("/api/workspaces")
    assert catalog.status_code == 200
    original_id = catalog.json()["active_workspace_id"]

    downloaded = client.get(f"/api/workspaces/{original_id}/backup")
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.headers["content-type"].startswith("application/zip")
    assert downloaded.headers["cache-control"] == "no-store"
    assert int(downloaded.headers["content-length"]) == len(downloaded.content)
    assert int(downloaded.headers["x-axiom-backup-files"]) == 3
    assert len(downloaded.headers["x-axiom-backup-manifest-sha256"]) == 64

    restored = client.post(
        "/api/workspaces/backup/restore?name=Restored%20workspace",
        content=downloaded.content,
        headers={"content-type": "application/zip"},
    )
    assert restored.status_code == 201, restored.text
    payload = restored.json()
    restored_id = payload["workspace"]["id"]
    assert restored_id != original_id
    assert payload["backup"]["integrity"] == "verified"
    assert payload["scope_note"].startswith("The backup was restored into a new managed workspace")

    registry = WorkspaceRegistry()
    restored_root = registry.resolve(restored_id).root
    assert (restored_root / "README.md").read_text(encoding="utf-8") == "original workspace"
    assert (restored_root / ".axiom" / "models.json").read_text(encoding="utf-8") == "[]\n"
    assert (restored_root / "data" / "private.jsonl").read_text(encoding="utf-8") == '{"row":"private"}\n'
    assert (default / "README.md").read_text(encoding="utf-8") == "original workspace"
    assert client.get("/api/workspaces").json()["active_workspace_id"] == original_id


def test_workspace_backup_api_rejects_invalid_archive_without_registering_workspace(tmp_path: Path, monkeypatch):
    client, default = _client(tmp_path, monkeypatch)
    original_id = client.get("/api/workspaces").json()["active_workspace_id"]
    existing = client.get("/api/workspaces")
    before_ids = {item["id"] for item in existing.json()["workspaces"]}

    rejected = client.post(
        "/api/workspaces/backup/restore?name=Rejected",
        content=b"not a zip",
        headers={"content-type": "application/zip"},
    )
    assert rejected.status_code == 422
    assert {item["id"] for item in client.get("/api/workspaces").json()["workspaces"]} == before_ids
    assert client.get("/api/workspaces").json()["active_workspace_id"] == original_id


def test_workspace_backup_api_requires_zip_media_type_and_registered_workspace(tmp_path: Path, monkeypatch):
    client, _default = _client(tmp_path, monkeypatch)
    workspace_id = client.get("/api/workspaces").json()["active_workspace_id"]
    invalid_type = client.post(
        "/api/workspaces/backup/restore?name=Wrong%20type",
        content=b"data",
        headers={"content-type": "text/plain"},
    )
    assert invalid_type.status_code == 415
    missing = client.get("/api/workspaces/ws-" + "0" * 32 + "/backup")
    assert missing.status_code == 404
