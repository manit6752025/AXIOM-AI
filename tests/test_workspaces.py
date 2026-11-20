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



def test_workflow_deployment_publishes_a_separate_non_executable_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    default = tmp_path / "project"
    default.mkdir()
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / "catalog"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / "managed"))
    monkeypatch.chdir(default)

    from axiom.api.server import create_app

    client = TestClient(create_app())
    source_workspace_id = client.get("/api/workspaces").json()["active_workspace_id"]
    target = client.post("/api/workspaces", json={"name": "Deployment target"})
    assert target.status_code == 201
    target_workspace_id = target.json()["workspace"]["id"]

    created = client.post(
        "/api/workflows",
        json={"name": "Dataset review", "nodes": [{"id": "dataset", "type": "dataset.profile"}], "edges": []},
    )
    assert created.status_code == 201
    source_workflow = created.json()["workflow"]

    deployed = client.post(
        f"/api/workflows/{source_workflow['id']}/deploy",
        json={"target_workspace_id": target_workspace_id},
    )
    assert deployed.status_code == 201
    payload = deployed.json()
    assert payload["deployment"] == {
        "target_workspace_id": target_workspace_id,
        "target_workspace_name": "Deployment target",
        "status": "published_copy",
        "executed": False,
    }
    assert payload["workflow"]["id"] != source_workflow["id"]
    assert payload["workflow"]["execution_status"] == "not_executable"
    assert payload["workflow"]["deployment"]["source_workspace_id"] == source_workspace_id
    assert payload["workflow"]["deployment"]["source_workflow_id"] == source_workflow["id"]
    assert payload["workflow"]["revision"] == 1
    assert payload["workflow"]["versions"] == []

    source_list = client.get("/api/workflows")
    target_list = client.get("/api/workflows", headers={"X-AXIOM-Workspace-ID": target_workspace_id})
    assert [item["id"] for item in source_list.json()["workflows"]] == [source_workflow["id"]]
    assert [item["id"] for item in target_list.json()["workflows"]] == [payload["workflow"]["id"]]


def test_workflow_deployment_rejects_same_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    default = tmp_path / "project"
    default.mkdir()
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / "catalog"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / "managed"))
    monkeypatch.chdir(default)

    from axiom.api.server import create_app

    client = TestClient(create_app())
    workspace_id = client.get("/api/workspaces").json()["active_workspace_id"]
    workflow = client.post(
        "/api/workflows",
        json={"name": "Local only", "nodes": [{"id": "dataset", "type": "dataset.profile"}], "edges": []},
    ).json()["workflow"]
    response = client.post(
        f"/api/workflows/{workflow['id']}/deploy",
        json={"target_workspace_id": workspace_id},
    )
    assert response.status_code == 409



def test_model_studio_storage_isolated_for_managed_workspaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from axiom.core.model_studio import native_model_root, studio_home
    from axiom.core.workspace_context import reset_workspace_context, set_workspace_context

    monkeypatch.delenv("AXIOM_MODEL_STUDIO_HOME", raising=False)
    workspace_root = tmp_path / "managed-workspace"
    workspace_root.mkdir()
    token = set_workspace_context(workspace_root, "ws-" + "a" * 32)
    try:
        assert studio_home() == workspace_root / ".axiom" / "model-studio"
        assert native_model_root() == workspace_root / "models" / "native"
    finally:
        reset_workspace_context(token)
