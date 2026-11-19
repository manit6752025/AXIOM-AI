from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import axiom.cli.main as cli_main

app = cli_main.app
from axiom.models.registry import Model, ModelRegistry


runner = CliRunner()


def test_model_list_json_has_stable_schema_and_empty_behavior(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    registry = ModelRegistry()
    registry.add(Model(name="tiny-llama", source="/models/tiny-llama", format="safetensors"))

    result = runner.invoke(app, ["model", "list", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["total"] == 1
    assert payload["models"][0]["name"] == "tiny-llama"

    empty_dir = tmp_path / "empty-project"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    result = runner.invoke(app, ["model", "list", "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["models"] == []


def test_model_info_json_is_bounded_and_does_not_download_weights(monkeypatch) -> None:
    def fake_model_info(*, repo_id: str, files_metadata: bool):
        assert files_metadata is True
        return SimpleNamespace(
            id=repo_id,
            author="org",
            library_name="transformers",
            pipeline_tag="text-generation",
            downloads=12,
            likes=3,
            gated=False,
            private=False,
            tags=["llama", "text-generation"],
            siblings=[
                SimpleNamespace(rfilename="model.safetensors"),
                SimpleNamespace(rfilename="model.gguf"),
            ],
        )

    monkeypatch.setattr(cli_main, "HfApi", lambda: SimpleNamespace(model_info=fake_model_info))
    result = runner.invoke(app, ["model", "info", "org/tiny", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["model"]["id"] == "org/tiny"
    assert payload["model"]["formats"] == ["GGUF", "Safetensors"]
    assert payload["model"]["files"]["total"] == 2
    assert payload["model"]["files"]["truncated"] is False
    assert "no model weights were downloaded" in payload["scope_note"].lower()


def test_model_inspect_json_reports_local_capabilities(tmp_path: Path) -> None:
    model = tmp_path / "tiny-llama"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"metadata inspection fixture")

    result = runner.invoke(app, ["model", "inspect", str(model), "--json"])

    assert result.exit_code == 0, result.stdout
    inspection = json.loads(result.stdout)["inspection"]
    assert inspection["path"] == str(model.resolve())
    assert inspection["model_type"] == "llama"
    assert inspection["has_safetensors"] is True
    assert "Text generation" in inspection["capabilities"]


def test_dataset_inspect_validate_and_stats_share_json_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"text":"first"}\n'
        'not-json\n'
        '\n',
        encoding="utf-8",
    )

    inspect = runner.invoke(app, ["dataset", "inspect", str(dataset), "--json"])
    stats = runner.invoke(app, ["dataset", "stats", str(dataset), "--json"])
    validate = runner.invoke(app, ["dataset", "validate", str(dataset), "--json"])

    assert inspect.exit_code == 0, inspect.stdout
    assert stats.exit_code == 0, stats.stdout
    assert validate.exit_code == 1, validate.stdout
    reports = [json.loads(result.stdout) for result in (inspect, stats, validate)]
    assert all(payload["schema_version"] == 1 for payload in reports)
    assert all(payload["dataset"]["invalid"] == 1 for payload in reports)
    assert all(payload["dataset"]["errors"][0]["line"] == 2 for payload in reports)


def test_integration_list_json_returns_empty_or_registered_records(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["integration", "list", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["total"] == 0
    assert payload["integrations"] == []


def test_model_search_json_skips_symlink_entries(tmp_path: Path) -> None:
    root = tmp_path / "models"
    model_dir = root / "tiny-llama"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    outside = tmp_path / "private.safetensors"
    outside.write_bytes(b"private")
    linked = root / "linked-llama.safetensors"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        # Windows and restricted containers may disable symlink creation. A
        # failed attempt must not leave a partial path behind.
        assert not linked.exists(), f"Failed symlink creation left an entry: {exc}"
    else:
        assert linked.is_symlink()

    result = runner.invoke(app, ["model", "search", "llama", "--path", str(root), "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["total"] >= 1
    assert all("path" in item and "kind" in item and "size_bytes" in item for item in payload["matches"])
    assert all("private.safetensors" not in item["path"] for item in payload["matches"])



def test_model_analyze_json_reports_metadata_and_estimate_basis(tmp_path: Path, monkeypatch) -> None:
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "torch_dtype": "bfloat16",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_model_info(*, repo_id: str, files_metadata: bool):
        assert files_metadata is True
        return SimpleNamespace(
            id=repo_id,
            siblings=[
                SimpleNamespace(rfilename="config.json", size=512),
                SimpleNamespace(rfilename="model.safetensors", size=4096),
            ],
        )

    monkeypatch.setattr(cli_main, "HfApi", lambda: SimpleNamespace(
        model_info=fake_model_info,
        hf_hub_download=lambda *, repo_id, filename: str(config_path),
    ))
    result = runner.invoke(app, ["model", "analyze", "org/tiny", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    analysis = payload["analysis"]
    assert payload["schema_version"] == 1
    assert analysis["repo_id"] == "org/tiny"
    assert analysis["architecture"] == "LlamaForCausalLM"
    assert analysis["context_length"] == 128
    assert analysis["parameter_source"] == "architecture estimate"
    assert analysis["files"]["items"] == ["config.json", "model.safetensors"]
    assert "no model weights were downloaded" in payload["scope_note"].lower()


def test_system_info_json_reports_hardware_and_fit_scope(monkeypatch) -> None:
    hardware = SimpleNamespace(
        os_name="Linux",
        architecture="x86_64",
        cpu_cores=8,
        ram_gb=16.0,
        gpu_name="Test GPU",
        vram_gb=8.0,
        cuda_available=True,
    )
    monkeypatch.setattr(cli_main, "detect_hardware", lambda: hardware)
    monkeypatch.setattr(cli_main, "estimate_model_fit", lambda billions, _hardware: f"fit-{billions}B")

    result = runner.invoke(app, ["system", "info", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["hardware"]["gpu_name"] == "Test GPU"
    assert payload["hardware"]["vram_gb"] == 8.0
    assert len(payload["model_fit_estimates"]) == 7
    assert payload["model_fit_estimates"][0]["assessment"] == "fit-1B"
    assert "not measured runtime capacity" in payload["fit_scope_note"]



def test_dataset_clean_json_reports_atomic_output_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    output = tmp_path / "cleaned.jsonl"
    original = '{"text":"first"}\n{"text":"first"}\nnot-json\n\n[1,2]\n'
    dataset.write_text(original, encoding="utf-8")

    result = runner.invoke(app, ["dataset", "clean", str(dataset), "--output", str(output), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    report = payload["cleaning"]
    assert payload["schema_version"] == 1
    assert report["total_lines"] == 5
    assert report["kept"] == 1
    assert report["removed_duplicates"] == 1
    assert report["removed_invalid"] == 1
    assert report["removed_empty"] == 1
    assert report["removed_non_objects"] == 1
    assert dataset.read_text(encoding="utf-8") == original
    assert output.read_text(encoding="utf-8") == '{"text": "first"}\n'


def test_dataset_clean_json_rejects_in_place_cleaning_without_touching_source(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    original = '{"text":"keep"}\nnot-json\n'
    dataset.write_text(original, encoding="utf-8")

    result = runner.invoke(app, ["dataset", "clean", str(dataset), "--output", str(dataset), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "dataset_cleaning_failed"
    assert dataset.read_text(encoding="utf-8") == original



def test_model_registry_export_import_and_remove_json_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first_add = runner.invoke(app, [
        "model", "add", "tiny-one", "/models/tiny-one", "--format", "safetensors", "--json",
    ])
    second_add = runner.invoke(app, [
        "model", "add", "tiny-two", "/models/tiny-two", "--format", "gguf", "--json",
    ])
    assert first_add.exit_code == 0, first_add.stdout
    assert second_add.exit_code == 0, second_add.stdout
    first_model = json.loads(first_add.stdout)["model"]
    assert json.loads(second_add.stdout)["created"] is True

    export_path = tmp_path / "exports" / "models.json"
    exported = runner.invoke(app, ["model", "export", "--path", str(export_path), "--json"])
    assert exported.exit_code == 0, exported.stdout
    export_payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert export_payload["schema_version"] == 1
    assert export_payload["total"] == 2
    assert json.loads(exported.stdout)["path"] == str(export_path.resolve())

    target_project = tmp_path / "target-project"
    target_project.mkdir()
    monkeypatch.chdir(target_project)
    imported = runner.invoke(app, ["model", "import", str(export_path), "--json"])
    assert imported.exit_code == 0, imported.stdout
    import_payload = json.loads(imported.stdout)
    assert import_payload["imported"] == 2
    assert import_payload["total_models_after_import"] == 2

    favorite = runner.invoke(app, ["model", "favorite", first_model["id"], "--unfavorite", "--json"])
    assert favorite.exit_code == 0, favorite.stdout
    assert json.loads(favorite.stdout)["model"]["favorite"] is False

    removed = runner.invoke(app, ["model", "remove", first_model["id"], "--json"])
    assert removed.exit_code == 0, removed.stdout
    assert json.loads(removed.stdout)["removed"] is True
    listed = runner.invoke(app, ["model", "list", "--json"])
    assert len(json.loads(listed.stdout)["models"]) == 1


def test_model_export_refuses_live_registry_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["model", "add", "tiny-one", "/models/tiny-one", "--format", "safetensors"])
    live_registry = tmp_path / ".axiom" / "models.json"
    before = live_registry.read_text(encoding="utf-8")
    result = runner.invoke(app, ["model", "export", "--path", str(live_registry), "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "model_export_failed"
    assert live_registry.read_text(encoding="utf-8") == before


def test_model_import_json_rejects_duplicate_batch_without_partial_import(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    existing = runner.invoke(app, ["model", "add", "already-here", "/models/already-here", "--format", "safetensors", "--json"])
    assert existing.exit_code == 0
    existing_model = json.loads(existing.stdout)["model"]
    imported_one = {**existing_model, "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "name": "new-one", "source": "/models/new-one"}
    duplicate = {**imported_one, "id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "name": "new-two", "source": "/models/new-two"}
    input_file = tmp_path / "bad-batch.json"
    # Duplicate source identities must reject the entire import before any write.
    duplicate["source"] = "/models/new-one"
    input_file.write_text(json.dumps({"schema_version": 1, "models": [imported_one, duplicate]}), encoding="utf-8")

    result = runner.invoke(app, ["model", "import", str(input_file), "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "model_import_failed"
    listed = runner.invoke(app, ["model", "list", "--json"])
    models = json.loads(listed.stdout)["models"]
    assert [item["name"] for item in models] == ["already-here"]



def test_model_add_and_update_json_preserve_metadata_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    created = runner.invoke(app, [
        "model", "add", "tiny-llama", "/models/tiny-llama",
        "--format", "safetensors", "--architecture", "LlamaForCausalLM",
        "--license", "apache-2.0", "--tag", "local", "--tag", "reviewed",
        "--content-sha256", "a" * 64, "--json",
    ])
    assert created.exit_code == 0, created.stdout
    model = json.loads(created.stdout)["model"]
    assert model["architecture"] == "LlamaForCausalLM"
    assert model["license"] == "apache-2.0"
    assert model["tags"] == ["local", "reviewed"]
    assert model["content_sha256"] == "a" * 64

    updated = runner.invoke(app, [
        "model", "update", model["id"],
        "--architecture", "MistralForCausalLM",
        "--license", "mit", "--tag", "quality-reviewed",
        "--favorite", "--json",
    ])
    assert updated.exit_code == 0, updated.stdout
    record = json.loads(updated.stdout)["model"]
    assert record["architecture"] == "MistralForCausalLM"
    assert record["license"] == "mit"
    assert record["tags"] == ["quality-reviewed"]
    assert record["favorite"] is True
    assert record["content_sha256"] == "a" * 64


def test_model_update_validation_failure_does_not_mutate_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    created = runner.invoke(app, ["model", "add", "tiny-llama", "/models/tiny-llama", "--format", "safetensors", "--json"])
    original = json.loads(created.stdout)["model"]
    invalid = runner.invoke(app, [
        "model", "update", original["id"], "--license", "x" * 121, "--json",
    ])
    assert invalid.exit_code == 1
    assert json.loads(invalid.stdout)["error"]["code"] == "model_update_failed"
    listed = runner.invoke(app, ["model", "list", "--json"])
    model_after = json.loads(listed.stdout)["models"][0]
    assert model_after == original
