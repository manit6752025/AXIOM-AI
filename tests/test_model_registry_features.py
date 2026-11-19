import json

import pytest

from axiom.models.discovery import discover_local_model_folders
from axiom.models.registry import Model, ModelRegistry


def test_model_registry_migrates_stable_ids_and_persists_metadata(tmp_path):
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    legacy_file = registry_root / "models.json"
    legacy_file.write_text(
        json.dumps([{"name": "Demo", "source": "org/demo", "format": "gguf"}]),
        encoding="utf-8",
    )
    registry = ModelRegistry(registry_root, create=False)

    first = registry.list()[0]
    second = registry.list()[0]
    assert first.id == second.id
    assert registry.update(first.id, {"favorite": True, "tags": ["local"]}) == Model(
        name="Demo",
        source="org/demo",
        format="gguf",
        id=first.id,
        favorite=True,
        tags=["local"],
    )
    assert json.loads(legacy_file.read_text(encoding="utf-8"))[0]["id"] == first.id


def test_model_registry_rejects_duplicate_source_and_content(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    registry.add(Model(name="First", source="org/demo", content_sha256="a" * 64))

    with pytest.raises(ValueError, match="source already registered"):
        registry.add(Model(name="Second", source="ORG/DEMO"))
    with pytest.raises(ValueError, match="content already registered"):
        registry.add(Model(name="Third", source="different/model", content_sha256="a" * 64))


def test_model_registry_import_is_validated_and_never_overwrites(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    first = Model(name="Existing", source="org/existing")
    registry.add(first)

    imported = registry.import_models([{
        "name": "Imported",
        "source": "org/imported",
        "format": "safetensors",
        "license": "Apache-2.0",
    }])

    assert imported[0].license == "Apache-2.0"
    assert {model.name for model in registry.list()} == {"Existing", "Imported"}
    with pytest.raises(ValueError, match="already exists"):
        registry.import_models([{"name": "Existing", "source": "org/other"}])
    assert registry.get(first.id) == first


def test_workspace_model_api_paginates_imports_updates_and_removes_metadata_only(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"keep these weights")

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/workspace/models",
            json={
                "name": "Demo model",
                "source": str(weights),
                "format": "safetensors",
                "architecture": "DemoForCausalLM",
                "parameters": "8B",
                "license": "Apache-2.0",
                "tags": ["local", "chat"],
            },
        )
        assert created.status_code == 201
        model = created.json()["model"]
        assert len(model["id"]) == 32
        assert model["architecture"] == "DemoForCausalLM"

        page = client.get("/api/workspace/models", params={"limit": 1, "offset": 0})
        assert page.status_code == 200
        assert page.json()["total"] == 1
        assert page.json()["has_more"] is False

        updated = client.patch(
            f"/api/workspace/models/{model['id']}",
            json={"favorite": True, "tags": ["local", "favorite"]},
        )
        assert updated.status_code == 200
        assert updated.json()["model"]["id"] == model["id"]
        assert updated.json()["model"]["favorite"] is True

        exported = client.get("/api/workspace/models/export")
        assert exported.status_code == 200
        assert exported.json()["schema_version"] == 1
        assert exported.json()["models"][0]["id"] == model["id"]

        imported = client.post(
            "/api/workspace/models/import",
            json={
                "schema_version": 1,
                "models": [{"name": "Second", "source": "org/second", "format": "gguf"}],
            },
        )
        assert imported.status_code == 201
        assert imported.json()["imported"] == 1

        removed = client.delete(f"/api/workspace/models/{model['id']}")
        assert removed.status_code == 200
        assert removed.json()["weights_deleted"] is False
        assert weights.read_bytes() == b"keep these weights"
        assert client.get(f"/api/workspace/models/{model['id']}").status_code == 404


def test_model_discovery_is_bounded_and_skips_symlinks(tmp_path):
    models_root = tmp_path / "models"
    candidate = models_root / "nested" / "demo"
    candidate.mkdir(parents=True)
    (candidate / "config.json").write_text('{"model_type":"demo"}', encoding="utf-8")
    (candidate / "weights.safetensors").write_bytes(b"weights")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.gguf").write_bytes(b"external")
    try:
        (models_root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    result = discover_local_model_folders(models_root, max_candidates=1)
    assert result["candidates"] == [{
        "path": "nested/demo",
        "format": "Safetensors",
        "has_config": True,
        "weight_file_count": 1,
        "weight_size_bytes": len(b"weights"),
    }]
    assert "linked" not in {item["path"] for item in result["candidates"]}


def test_workspace_model_discovery_and_inspection_use_trusted_paths(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    model_dir = tmp_path / "models" / "demo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "demo", "architectures": ["DemoForCausalLM"]}),
        encoding="utf-8",
    )
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        discovered = client.get("/api/workspace/models/discover")
        assert discovered.status_code == 200
        assert discovered.json()["candidates"][0]["path"] == "demo"

        inspected = client.post("/api/workspace/models/inspect", json={"path": "demo"})
        assert inspected.status_code == 200
        assert inspected.json()["inspection"]["architecture"] == "DemoForCausalLM"
        assert inspected.json()["inspection"]["model_format"] == "Safetensors"

        unsafe = client.post("/api/workspace/models/inspect", json={"path": "../../outside"})
        assert unsafe.status_code == 422
