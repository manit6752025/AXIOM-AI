from __future__ import annotations

import io
import json
import zipfile

import pytest
from typer.testing import CliRunner

from axiom.core.agent_profiles import AgentProfileStore
from axiom.core.experiments import ExperimentStore
from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType
from axiom.core.tool_permissions import ToolPermissionStore
from axiom.core.workflows import WorkflowStore
from axiom.evaluation.store import EvaluationStore
from axiom.models.registry import Model, ModelRegistry
from axiom.core.workspace_metadata import (
    ARCHIVE_FORMAT,
    export_workspace_metadata,
    import_workspace_metadata,
)
from axiom.cli.main import app


runner = CliRunner()


def _populate_workspace(root):
    metadata = root / ".axiom"
    metadata.mkdir(parents=True, exist_ok=True)
    ModelRegistry(metadata).add(Model(name="Local Qwen", source="models/qwen", format="safetensors"))
    ExperimentStore(metadata).add({
        "id": "experiment-one",
        "schema_version": 1,
        "name": "Baseline run",
        "tags": ["baseline"],
    })
    WorkflowStore(metadata).save({
        "id": "workflow-one",
        "name": "Inspect model",
        "nodes": [{"id": "inspect", "type": "model.inspect"}],
        "edges": [],
    })
    EvaluationStore(metadata).add({
        "id": "evaluation-one",
        "schema_version": 1,
        "metrics": {"exact_match": 0.75},
    })
    AgentProfileStore(root).create("model-review", ["model.inspect"])
    IntegrationRegistry(metadata).add(AgentIntegration(
        name="private-agent",
        type=IntegrationType.HTTP,
        url="https://agent.example.test/?token=do-not-export",
        description="Private endpoint",
    ))
    ToolPermissionStore(metadata).save_profile("trusted", ["axiom_system_info"])
    ToolPermissionStore(metadata).set_active("trusted")
    (metadata / "prediction_jobs.json").write_text('{"jobs":[{"id":"do-not-export"}]}', encoding="utf-8")
    (root / "axiom.yaml").write_text("api_key: DO_NOT_EXPORT_CONFIG_SECRET\n", encoding="utf-8")
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "train.jsonl").write_text('{"text":"DO_NOT_EXPORT_DATASET"}\n', encoding="utf-8")


def test_workspace_metadata_archive_round_trip_is_allowlisted_and_validated(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _populate_workspace(source)

    archive = export_workspace_metadata(source)
    assert archive.startswith(b"PK")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = set(bundle.namelist())
        assert "manifest.json" in names
        assert names == {
            "manifest.json",
            "metadata/models.json",
            "metadata/experiments.json",
            "metadata/workflows.json",
            "metadata/evaluations.json",
            "metadata/agent-profiles.json",
        }
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["format"] == ARCHIVE_FORMAT
        assert manifest["schema_version"] == 1
        serialized = b"".join(bundle.read(name) for name in names if name != "manifest.json")
        for secret in (
            b"DO_NOT_EXPORT_CONFIG_SECRET",
            b"DO_NOT_EXPORT_DATASET",
            b"do-not-export",
            b"private-agent",
            b"trusted",
        ):
            assert secret not in serialized

    result = import_workspace_metadata(archive, destination)
    assert result["mode"] == "merge"
    assert result["imported"] == {
        "models": 1,
        "experiments": 1,
        "workflows": 1,
        "evaluations": 1,
        "agent_profiles": 2,
    }
    assert [item.name for item in ModelRegistry(destination / ".axiom").list()] == ["Local Qwen"]
    assert ExperimentStore(destination / ".axiom").get("experiment-one")["name"] == "Baseline run"
    assert WorkflowStore(destination / ".axiom").get("workflow-one")["name"] == "Inspect model"
    assert EvaluationStore(destination / ".axiom").get("evaluation-one")["id"] == "evaluation-one"
    assert {item["name"] for item in AgentProfileStore(destination).load()["profiles"]} == {"default", "model-review"}
    assert not (destination / "axiom.yaml").exists()
    assert not (destination / "data" / "train.jsonl").exists()
    assert not (destination / ".axiom" / "tool_permissions.json").exists()
    assert not (destination / ".axiom" / "prediction_jobs.json").exists()


def test_workspace_metadata_import_merges_existing_records_without_overwriting(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _populate_workspace(source)
    ModelRegistry(destination / ".axiom").add(Model(name="Existing", source="models/existing"))
    ExperimentStore(destination / ".axiom").add({
        "id": "existing-experiment",
        "schema_version": 1,
        "name": "Keep me",
        "tags": [],
    })

    result = import_workspace_metadata(export_workspace_metadata(source), destination)
    assert result["mode"] == "merge"
    assert {item.name for item in ModelRegistry(destination / ".axiom").list()} == {"Existing", "Local Qwen"}
    assert ExperimentStore(destination / ".axiom").get("existing-experiment") is not None
    assert ExperimentStore(destination / ".axiom").get("experiment-one") is not None


def test_workspace_metadata_replace_only_replaces_collections_in_archive(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _populate_workspace(source)
    archive = export_workspace_metadata(source, collections=["models"])
    ModelRegistry(destination / ".axiom").add(Model(name="Existing", source="models/existing"))
    ExperimentStore(destination / ".axiom").add({
        "id": "existing-experiment",
        "schema_version": 1,
        "name": "Keep me",
        "tags": [],
    })

    result = import_workspace_metadata(archive, destination, replace=True)
    assert result["mode"] == "replace"
    assert [item.name for item in ModelRegistry(destination / ".axiom").list()] == ["Local Qwen"]
    assert ExperimentStore(destination / ".axiom").get("existing-experiment") is not None


def test_workspace_metadata_conflict_rejects_all_changes_before_writing(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _populate_workspace(source)
    archive = export_workspace_metadata(source)
    ModelRegistry(destination / ".axiom").add(Model(name="Existing", source="models/existing"))
    EvaluationStore(destination / ".axiom").add({
        "id": "evaluation-one",
        "schema_version": 1,
        "metrics": {"exact_match": 0.1},
    })
    before_models = (destination / ".axiom" / "models.json").read_bytes()

    with pytest.raises(ValueError, match="conflicts"):
        import_workspace_metadata(archive, destination)

    assert (destination / ".axiom" / "models.json").read_bytes() == before_models
    assert [item.name for item in ModelRegistry(destination / ".axiom").list()] == ["Existing"]
    assert ExperimentStore(destination / ".axiom").list(1000) == []


def test_workspace_metadata_import_rejects_checksum_tampering_and_zip_traversal(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    _populate_workspace(root)
    original = export_workspace_metadata(root)

    tampered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original), "r") as source, zipfile.ZipFile(tampered, "w") as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename == "metadata/models.json":
                payload += b" "
            target.writestr(item, payload)
    with pytest.raises(ValueError, match="checksum"):
        import_workspace_metadata(tampered.getvalue(), root)

    traversal = io.BytesIO()
    with zipfile.ZipFile(traversal, "w") as target:
        target.writestr("../../outside.txt", b"unsafe")
    with pytest.raises(ValueError, match="unsupported or unsafe"):
        import_workspace_metadata(traversal.getvalue(), root)
    assert not (tmp_path / "outside.txt").exists()


def test_workspace_metadata_import_rejects_symlinked_metadata_directory(tmp_path):
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    archive = export_workspace_metadata(root)
    (root / ".axiom").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        import_workspace_metadata(archive, root)


def test_workspace_metadata_cli_export_and_import_round_trip(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _populate_workspace(source)
    archive = tmp_path / "workspace.zip"

    exported = runner.invoke(app, [
        "workspace", "metadata", "export",
        "--workspace", str(source),
        "--output", str(archive),
        "--json",
    ])
    assert exported.exit_code == 0, exported.output
    assert json.loads(exported.stdout)["collections"]["models"] == 1

    imported = runner.invoke(app, [
        "workspace", "metadata", "import",
        str(archive),
        "--workspace", str(destination),
        "--json",
    ])
    assert imported.exit_code == 0, imported.output
    assert json.loads(imported.stdout)["mode"] == "merge"
    assert len(ModelRegistry(destination / ".axiom").list()) == 1


def test_workspace_metadata_api_export_and_import(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from axiom.api.server import create_app

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _populate_workspace(source)
    monkeypatch.chdir(source)

    with TestClient(create_app()) as client:
        exported = client.get("/api/workspace/metadata/export")
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith(ARCHIVE_FORMAT)
        archive = exported.content

    monkeypatch.chdir(destination)
    with TestClient(create_app()) as client:
        imported = client.post(
            "/api/workspace/metadata/import",
            content=archive,
            headers={"Content-Type": ARCHIVE_FORMAT},
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["mode"] == "merge"

    assert len(ModelRegistry(destination / ".axiom").list()) == 1
