from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from axiom.core.workspaces import WorkspaceCatalogError, WorkspaceRegistry


def test_workspace_registry_preserves_default_and_isolates_new_roots(tmp_path: Path):
    default = tmp_path / "project"
    default.mkdir()
    legacy = default / ".axiom"
    legacy.mkdir()
    (legacy / "models.json").write_text("[]\n", encoding="utf-8")

    manager = WorkspaceRegistry(home=tmp_path / "catalog", workspaces_root=tmp_path / "managed", default_root=default)
    initial = manager.list_workspaces()
    assert len(initial) == 1
    default_id = initial[0]["id"]
    assert initial[0]["is_default"] is True
    assert manager.resolve(default_id).root == default
    assert (legacy / "models.json").read_text(encoding="utf-8") == "[]\n"

    created = manager.create("Inference lab")
    created_id = created["id"]
    created_root = manager.resolve(created_id).root
    assert created_root != default
    assert created_root.is_relative_to((tmp_path / "managed").resolve())
    assert created["is_default"] is False
    assert all("root" not in item and "path" not in item for item in manager.list_workspaces())

    renamed = manager.rename(created_id, "Production models")
    assert renamed["name"] == "Production models"
    assert manager.resolve(created_id).root == created_root
    assert manager.resolve(default_id).root == default


def test_workspace_registry_rejects_duplicate_names_and_unknown_ids(tmp_path: Path):
    default = tmp_path / "project"
    default.mkdir()
    manager = WorkspaceRegistry(home=tmp_path / "catalog", workspaces_root=tmp_path / "managed", default_root=default)
    manager.create("Evaluation")
    with pytest.raises(WorkspaceCatalogError, match="already exists"):
        manager.create(" evaluation ")
    with pytest.raises(KeyError):
        manager.resolve("ws-" + "0" * 32)


def test_workspace_catalog_rejects_symlinked_managed_root(tmp_path: Path):
    default = tmp_path / "project"
    default.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    link = tmp_path / "managed-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    manager = WorkspaceRegistry(home=tmp_path / "catalog", workspaces_root=link, default_root=default)
    with pytest.raises(WorkspaceCatalogError, match="symbolic link"):
        manager.create("Unsafe")


def test_workspace_api_scopes_model_registries_per_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    default = tmp_path / "project"
    default.mkdir()
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / "catalog"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / "managed"))
    monkeypatch.chdir(default)

    from axiom.api.server import create_app

    client = TestClient(create_app())
    initial = client.get("/api/workspaces")
    assert initial.status_code == 200
    first_id = initial.json()["active_workspace_id"]
    created = client.post("/api/workspaces", json={"name": "Isolated models"})
    assert created.status_code == 201
    second_id = created.json()["workspace"]["id"]

    added = client.post("/api/workspace/models", json={"name": "Default model", "source": "models/default"})
    assert added.status_code == 201
    second = client.post(
        "/api/workspace/models",
        headers={"X-AXIOM-Workspace-ID": second_id},
        json={"name": "Private model", "source": "models/private"},
    )
    assert second.status_code == 201

    default_models = client.get("/api/workspace/models")
    isolated_models = client.get("/api/workspace/models", headers={"X-AXIOM-Workspace-ID": second_id})
    assert [item["name"] for item in default_models.json()["models"]] == ["Default model"]
    assert [item["name"] for item in isolated_models.json()["models"]] == ["Private model"]

    renamed = client.patch(f"/api/workspaces/{second_id}", json={"name": "Renamed lab"})
    assert renamed.status_code == 200
    assert renamed.json()["workspace"]["name"] == "Renamed lab"
    assert first_id != second_id


def test_workspace_api_rejects_unknown_workspace_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    default = tmp_path / "project"
    default.mkdir()
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / "catalog"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / "managed"))
    monkeypatch.chdir(default)

    from axiom.api.server import create_app

    client = TestClient(create_app())
    response = client.get("/api/workspace/models", headers={"X-AXIOM-Workspace-ID": "ws-" + "0" * 32})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "workspace_not_found"
