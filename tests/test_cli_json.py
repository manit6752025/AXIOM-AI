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


def test_integration_trust_json_records_explicit_trust_and_unverified_owner_label(tmp_path: Path, monkeypatch) -> None:
    from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType

    monkeypatch.chdir(tmp_path)
    IntegrationRegistry().add(
        AgentIntegration("Local MCP", IntegrationType.MCP, command="python -m local.server")
    )

    result = runner.invoke(
        app,
        ["integration", "trust", "Local MCP", "trusted", "--owner-type", "workspace", "--owner-label", "Research Team", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["trust_status"] == "trusted"
    assert payload["owner_type"] == "workspace"
    assert payload["owner_label"] == "Research Team"
    assert payload["ownership_verified"] is False
    saved = IntegrationRegistry().list()[0]
    assert saved.trust_status == "trusted"
    assert saved.owner_label == "Research Team"

    trust_only = runner.invoke(app, ["integration", "trust", "Local MCP", "blocked", "--json"])
    assert trust_only.exit_code == 0, trust_only.stdout
    preserved = json.loads(trust_only.stdout)
    assert preserved["trust_status"] == "blocked"
    assert preserved["owner_type"] == "workspace"
    assert preserved["owner_label"] == "Research Team"


def test_integration_mcp_list_json_is_passive_and_redacts_command_arguments(tmp_path: Path, monkeypatch) -> None:
    from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType

    monkeypatch.chdir(tmp_path)
    registry = IntegrationRegistry()
    registry.add(AgentIntegration("Private MCP", IntegrationType.MCP, command="python -m private.server --token secret-value"))
    registry.add(AgentIntegration("Local CLI", IntegrationType.CLI, command="tool"))

    result = runner.invoke(app, ["integration", "mcp-list", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["total"] == 1
    assert payload["servers"][0]["name"] == "Private MCP"
    assert payload["servers"][0]["executable"] == "python"
    assert payload["servers"][0]["connection_state"] == "not_inspected"
    assert "secret-value" not in result.stdout
    assert "private.server" not in result.stdout


def test_integration_handshake_json_reports_protocol_metadata(tmp_path: Path, monkeypatch) -> None:
    from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType
    from axiom.core import mcp_inventory

    monkeypatch.chdir(tmp_path)
    IntegrationRegistry().add(
        AgentIntegration("Local MCP", IntegrationType.MCP, command="python -m local.server")
    )

    async def fake_inspect(name: str):
        assert name == "Local MCP"
        return {
            "name": name,
            "transport": "stdio",
            "status": "ok",
            "protocol_version": "2025-11-25",
            "server_name": "Local test server",
            "server_version": "1.0",
            "capabilities": {"tools": True, "resources": False, "prompts": False, "logging": False, "completions": False},
            "duration_ms": 12,
            "tool_execution_performed": False,
            "schemas_enumerated": False,
        }

    monkeypatch.setattr(mcp_inventory, "inspect_mcp_server", fake_inspect)
    result = runner.invoke(app, ["integration", "handshake", "Local MCP", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["server_name"] == "Local test server"
    assert payload["tool_execution_performed"] is False
    assert payload["schemas_enumerated"] is False


def test_integration_schemas_json_returns_bounded_read_only_metadata(tmp_path: Path, monkeypatch) -> None:
    from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType
    from axiom.core import mcp_inventory

    monkeypatch.chdir(tmp_path)
    IntegrationRegistry().add(
        AgentIntegration("Local MCP", IntegrationType.MCP, command="python -m local.server")
    )

    async def fake_inspect(name: str):
        assert name == "Local MCP"
        return {
            "name": name,
            "transport": "stdio",
            "status": "ok",
            "protocol_version": "2025-11-25",
            "tools": [{
                "name": "lookup",
                "title": "Lookup",
                "description": "Find records",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "output_schema": None,
                "annotations": {"readOnlyHint": True},
            }],
            "resources": [],
            "resource_templates": [],
            "truncated": {"tools": False, "resources": False, "resource_templates": False},
            "duration_ms": 9,
            "tool_execution_performed": False,
            "resource_read_performed": False,
            "prompt_execution_performed": False,
        }

    monkeypatch.setattr(mcp_inventory, "inspect_mcp_schemas", fake_inspect)
    result = runner.invoke(app, ["integration", "schemas", "Local MCP", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["tools"][0]["name"] == "lookup"
    assert payload["tool_execution_performed"] is False
    assert payload["resource_read_performed"] is False


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

    removed = runner.invoke(app, ["model", "remove", first_model["id"], "--yes", "--json"])
    assert removed.exit_code == 0, removed.stdout
    assert json.loads(removed.stdout)["removed"] is True
    assert json.loads(removed.stdout)["weights_deleted"] is False
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
        "--content-sha256", "a" * 64, "--size-bytes", "4096",
        "--workspace-label", "production", "--workspace-label", "team-a", "--json",
    ])
    assert created.exit_code == 0, created.stdout
    model = json.loads(created.stdout)["model"]
    assert model["architecture"] == "LlamaForCausalLM"
    assert model["license"] == "apache-2.0"
    assert model["tags"] == ["local", "reviewed"]
    assert model["content_sha256"] == "a" * 64
    assert model["size_bytes"] == 4096
    assert model["workspace_labels"] == ["production", "team-a"]

    updated = runner.invoke(app, [
        "model", "update", model["id"],
        "--architecture", "MistralForCausalLM",
        "--license", "mit", "--tag", "quality-reviewed", "--size-bytes", "8192",
        "--workspace-label", "staging", "--favorite", "--json",
    ])
    assert updated.exit_code == 0, updated.stdout
    record = json.loads(updated.stdout)["model"]
    assert record["architecture"] == "MistralForCausalLM"
    assert record["license"] == "mit"
    assert record["tags"] == ["quality-reviewed"]
    assert record["favorite"] is True
    assert record["content_sha256"] == "a" * 64
    assert record["size_bytes"] == 8192
    assert record["workspace_labels"] == ["staging"]

    cleared = runner.invoke(app, [
        "model", "update", model["id"], "--clear-workspace-labels", "--json",
    ])
    assert cleared.exit_code == 0, cleared.stdout
    assert json.loads(cleared.stdout)["model"]["workspace_labels"] == []


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



def test_model_add_json_rejects_duplicate_source_and_content_without_partial_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    created = runner.invoke(app, [
        "model", "add", "first-model", "org/demo-model",
        "--content-sha256", "a" * 64, "--json",
    ])
    assert created.exit_code == 0, created.stdout

    same_source = runner.invoke(app, [
        "model", "add", "same-source", "ORG/DEMO-MODEL", "--json",
    ])
    assert same_source.exit_code == 1
    assert json.loads(same_source.stdout)["error"]["code"] == "model_add_failed"
    assert "source already registered" in json.loads(same_source.stdout)["error"]["message"]

    same_content = runner.invoke(app, [
        "model", "add", "same-content", "org/another-location",
        "--content-sha256", "A" * 64, "--json",
    ])
    assert same_content.exit_code == 1
    assert json.loads(same_content.stdout)["error"]["code"] == "model_add_failed"
    assert "content already registered" in json.loads(same_content.stdout)["error"]["message"]

    listed = runner.invoke(app, ["model", "list", "--json"])
    payload = json.loads(listed.stdout)
    assert payload["total"] == 1
    assert [item["name"] for item in payload["models"]] == ["first-model"]


def test_model_add_and_update_json_include_source_attribution_and_license_review(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    created = runner.invoke(app, [
        "model", "add", "attributed-model", "org/attributed-model",
        "--license", "Apache-2.0",
        "--license-review", "unreviewed",
        "--source-attribution", "Example Research",
        "--source-url", "https://huggingface.co/example/model",
        "--source-revision", "0123456789abcdef",
        "--license-review-notes", "Pending manual review.",
        "--json",
    ])
    assert created.exit_code == 0, created.stdout
    model = json.loads(created.stdout)["model"]
    assert model["source_attribution"] == "Example Research"
    assert model["source_url"] == "https://huggingface.co/example/model"
    assert model["source_revision"] == "0123456789abcdef"
    assert model["license_review"] == "unreviewed"
    assert model["license_review_notes"] == "Pending manual review."

    updated = runner.invoke(app, [
        "model", "update", model["id"],
        "--license-review", "restricted",
        "--license-review-notes", "Distribution needs additional review.",
        "--source-revision", "fedcba9876543210",
        "--json",
    ])
    assert updated.exit_code == 0, updated.stdout
    record = json.loads(updated.stdout)["model"]
    assert record["license_review"] == "restricted"
    assert record["license_review_notes"] == "Distribution needs additional review."
    assert record["source_revision"] == "fedcba9876543210"



def test_model_remove_requires_confirmation_and_never_deletes_weight_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "models" / "tiny-model.gguf"
    weights.parent.mkdir()
    weights.write_bytes(b"model-weight-fixture")
    registry = ModelRegistry()
    model = Model(name="Tiny model", source=str(weights), format="gguf")
    registry.add(model)

    refused = runner.invoke(app, ["model", "remove", model.id, "--json"])
    assert refused.exit_code == 1
    refused_payload = json.loads(refused.stdout)
    assert refused_payload["error"]["code"] == "model_remove_confirmation_required"
    assert registry.get(model.id) is not None
    assert weights.read_bytes() == b"model-weight-fixture"

    cancelled = runner.invoke(app, ["model", "remove", model.id])
    assert cancelled.exit_code == 0
    assert "Cancelled" in cancelled.stdout
    assert registry.get(model.id) is not None
    assert weights.exists()

    removed = runner.invoke(app, ["model", "remove", model.id, "--yes", "--json"])
    assert removed.exit_code == 0, removed.stdout
    payload = json.loads(removed.stdout)
    assert payload["removed"] is True
    assert payload["weights_deleted"] is False
    assert payload["model"]["id"] == model.id
    assert registry.get(model.id) is None
    assert weights.read_bytes() == b"model-weight-fixture"



def test_dataset_profile_json_is_versioned_privacy_safe_and_preserves_source(tmp_path: Path) -> None:
    dataset = tmp_path / "private.jsonl"
    original = '{"email":"private.person@example.com","text":"private row content"}\nnot-json\n'
    dataset.write_text(original, encoding="utf-8")

    result = runner.invoke(app, ["dataset", "profile", str(dataset), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["profile"]["valid_rows"] == 1
    assert payload["profile"]["invalid_rows"] == 1
    assert payload["profile"]["privacy_review_recommended"] is True
    assert "private.person@example.com" not in result.stdout
    assert "private row content" not in result.stdout
    assert "matched PII values are never returned" in payload["scope_note"]
    assert dataset.read_text(encoding="utf-8") == original



def test_ai_huggingface_search_json_is_versioned_and_metadata_only(monkeypatch) -> None:
    from axiom.cli import ai_features

    observed = {}
    def fake_search(query: str, limit: int):
        observed.update(query=query, limit=limit)
        return [{"id": "org/tiny-qwen", "downloads": 123, "likes": 7, "pipeline_tag": "text-generation"}]

    monkeypatch.setattr(ai_features, "search_models", fake_search)
    result = runner.invoke(app, ["ai", "hf-search", "qwen", "--limit", "3", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert observed == {"query": "qwen", "limit": 3}
    assert payload["schema_version"] == 1
    assert payload["provider"] == "huggingface"
    assert payload["results"][0]["id"] == "org/tiny-qwen"
    assert "no model weights were downloaded" in payload["scope_note"]


def test_ai_policy_audit_json_has_stable_envelope_and_audit_only_scope(monkeypatch) -> None:
    from axiom.cli import ai_features

    monkeypatch.setattr(ai_features, "audit_model", lambda path: {
        "model": path,
        "status": "review_required",
        "policy_indicators": ["config.json: contains 'safety'"],
        "config_files": ["config.json"],
        "scope": "Audit only.",
    })
    result = runner.invoke(app, ["ai", "policy-audit", "models/test", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["audit"]["status"] == "review_required"
    assert payload["audit"]["policy_indicators"] == ["config.json: contains 'safety'"]
    assert "does not remove" in payload["scope_note"]


def test_ai_optimization_plan_json_uses_versioned_plan_contract(monkeypatch) -> None:
    from axiom.cli import ai_features

    class FakePlan:
        def to_dict(self):
            return {
                "model": "models/tiny",
                "estimated_memory_gb": 2.5,
                "recommended_quantization": "int4",
                "fits_estimate": True,
                "notes": ["Estimate only."],
            }

    monkeypatch.setattr(ai_features, "plan_optimization", lambda **_kwargs: FakePlan())
    result = runner.invoke(app, ["ai", "plan", "--model", "models/tiny", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["plan"]["model"] == "models/tiny"
    assert payload["plan"]["recommended_quantization"] == "int4"
    assert "must be measured" in payload["scope_note"]



def test_project_validate_json_reports_missing_paths_and_nonzero_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "axiom.yaml").write_text("name: test\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    result = runner.invoke(app, ["project", "validate", "--json"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["valid"] is False
    assert payload["root"] == str(tmp_path)
    assert payload["missing_paths"] == ["models", "experiments", "evaluations", "outputs"]


def test_project_validate_json_accepts_complete_project(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for item in ("axiom.yaml", "README.md"):
        (tmp_path / item).write_text("test\n", encoding="utf-8")
    for item in ("data", "models", "experiments", "evaluations", "outputs"):
        (tmp_path / item).mkdir()
    result = runner.invoke(app, ["project", "validate", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["valid"] is True
    assert payload["missing_paths"] == []


def test_check_json_distinguishes_non_project_and_ready_project(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["check", "--json"])
    assert result.exit_code == 0, result.stdout
    empty = json.loads(result.stdout)
    assert empty["schema_version"] == 1
    assert empty["status"] == "not_project"
    assert empty["project_detected"] is False
    assert "axiom.yaml" in empty["missing_paths"]

    (tmp_path / "axiom.yaml").write_text("name: test\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    for item in ("data", "models", "experiments", "evaluations", "outputs"):
        (tmp_path / item).mkdir()
    result = runner.invoke(app, ["check", "--json"])
    assert result.exit_code == 0, result.stdout
    ready = json.loads(result.stdout)
    assert ready["status"] == "ready"
    assert ready["project_detected"] is True
    assert ready["missing_paths"] == []



def test_config_show_json_redacts_nested_credentials_but_preserves_ordinary_values(tmp_path: Path) -> None:
    config = tmp_path / "axiom.yaml"
    config.write_text(
        "name: test\n"
        "max_tokens: 128\n"
        "api_key: private-api-key-value\n"
        "provider:\n"
        "  client_secret: private-client-secret\n"
        "  auth:\n"
        "    bearer_token: private-bearer-token\n"
        "  temperature: 0.2\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["config", "show", "--path", str(config), "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["redacted_value_count"] == 3
    assert payload["config"]["max_tokens"] == 128
    assert payload["config"]["api_key"] == "[REDACTED]"
    assert payload["config"]["provider"]["client_secret"] == "[REDACTED]"
    assert payload["config"]["provider"]["auth"]["bearer_token"] == "[REDACTED]"
    assert payload["config"]["provider"]["temperature"] == 0.2
    for secret in ("private-api-key-value", "private-client-secret", "private-bearer-token"):
        assert secret not in result.stdout


def test_config_show_json_rejects_symlinked_config_file(tmp_path: Path) -> None:
    target = tmp_path / "real.yaml"
    target.write_text("name: safe\n", encoding="utf-8")
    link = tmp_path / "linked.yaml"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        import pytest
        pytest.skip(f"File symlinks unavailable: {exc}")
    result = runner.invoke(app, ["config", "show", "--path", str(link), "--json"])
    assert result.exit_code != 0
    assert "could not be read safely" in result.output


def test_config_show_json_rejects_oversized_input_before_parsing(tmp_path: Path) -> None:
    config = tmp_path / "large.yaml"
    config.write_text("value: " + ("x" * (256 * 1024)), encoding="utf-8")
    result = runner.invoke(app, ["config", "show", "--path", str(config), "--json"])
    assert result.exit_code != 0
    assert "256 KiB" in result.output



def test_info_json_reports_local_environment_without_collecting_secrets(tmp_path: Path, monkeypatch) -> None:
    from axiom.cli import extended as extended_cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "axiom.yaml").write_text("name: local-test\n", encoding="utf-8")
    monkeypatch.setattr(extended_cli, "AXIOM_VERSION", "0.9.0-test")
    monkeypatch.setattr(extended_cli, "package_version", lambda: "0.9.0")
    monkeypatch.setattr(extended_cli.platform, "python_version", lambda: "3.12.1")
    monkeypatch.setattr(extended_cli.platform, "platform", lambda: "TestOS-1")
    monkeypatch.setattr(extended_cli.sys, "executable", "/usr/bin/python-test")
    monkeypatch.setattr(extended_cli, "git_output", lambda *args: "implement/roadmap")
    monkeypatch.setenv("AXIOM_API_TOKEN", "never-return-this-token")

    result = runner.invoke(app, ["info", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["version"] == "0.9.0-test"
    assert payload["package_version"] == "0.9.0"
    assert payload["python_version"] == "3.12.1"
    assert payload["platform"] == "TestOS-1"
    assert payload["executable"] == "/usr/bin/python-test"
    assert payload["working_directory"] == str(tmp_path)
    assert payload["project_config_present"] is True
    assert payload["git_branch"] == "implement/roadmap"
    assert "never-return-this-token" not in result.stdout
    assert "AXIOM_API_TOKEN" not in result.stdout



def test_huggingface_status_json_reports_unauthenticated_and_authenticated_without_tokens(monkeypatch) -> None:
    class LocalTokenNotFoundError(Exception):
        pass

    from axiom.cli import main as cli

    monkeypatch.setattr(cli, "whoami", lambda: (_ for _ in ()).throw(LocalTokenNotFoundError()))
    unauthenticated = runner.invoke(app, ["hf", "status", "--json"])
    assert unauthenticated.exit_code == 0, unauthenticated.stdout
    payload = json.loads(unauthenticated.stdout)
    assert payload["schema_version"] == 1
    assert payload["status"] == "not_authenticated"
    assert payload["authenticated"] is False

    monkeypatch.setattr(cli, "whoami", lambda: {
        "name": "test-account",
        "orgs": [{"name": "research-team"}, "open-source"],
        "token": "do-not-return-token",
    })
    authenticated = runner.invoke(app, ["hf", "status", "--json"])
    assert authenticated.exit_code == 0, authenticated.stdout
    payload = json.loads(authenticated.stdout)
    assert payload["schema_version"] == 1
    assert payload["authenticated"] is True
    assert payload["account"] == "test-account"
    assert payload["organizations"] == ["research-team", "open-source"]
    assert "do-not-return-token" not in authenticated.stdout


def test_supercompress_status_json_redacts_credential_and_reports_local_configuration(monkeypatch) -> None:
    from axiom.cli import main as cli

    monkeypatch.setenv("SUPERCOMPRESS_API_KEY", "private-supercompress-token")
    monkeypatch.setattr(cli, "redacted_base_url", lambda: "https://api.example.invalid/v1")
    result = runner.invoke(app, ["supercompress", "status", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["configured"] is True
    assert payload["credential_reference_available"] is True
    assert payload["credential_value_returned"] is False
    assert payload["endpoint"] == "https://api.example.invalid/v1"
    assert "private-supercompress-token" not in result.stdout
    assert "SUPERCOMPRESS_API_KEY" not in result.stdout


def test_integration_doctor_json_reports_availability_without_commands_or_urls(tmp_path: Path, monkeypatch) -> None:
    from axiom import cli as _cli_package  # keep module import resolution explicit
    from axiom.cli import main as cli
    from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType

    monkeypatch.chdir(tmp_path)
    registry = IntegrationRegistry()
    registry.add(AgentIntegration(
        "Local command",
        IntegrationType.CLI,
        command="python --token do-not-return-command-token",
    ))
    registry.add(AgentIntegration(
        "Remote endpoint",
        IntegrationType.HTTP,
        url="https://user:private@example.invalid/api?token=do-not-return-url-token",
    ))
    registry.add(AgentIntegration("Incomplete", IntegrationType.MCP))
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/python" if name == "python" else None)

    result = runner.invoke(app, ["integration", "doctor", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["total"] == 3
    records = {item["name"]: item for item in payload["integrations"]}
    assert records["Local command"]["status"] == "available"
    assert records["Local command"]["executable"] == "python"
    assert records["Remote endpoint"]["status"] == "configured"
    assert records["Remote endpoint"]["url_configured"] is True
    assert records["Incomplete"]["status"] == "incomplete"
    for secret in ("do-not-return-command-token", "do-not-return-url-token", "private"):
        assert secret not in result.stdout
    assert "No command is executed" in payload["scope_note"]
