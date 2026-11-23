from __future__ import annotations

import json

from typer.testing import CliRunner

import axiom.cli.engine as engine_cli
from axiom.core.engine_fabric_profiles import EngineFabricProfileStore

runner = CliRunner()


def test_engine_fabric_profile_cli_save_list_connect_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine_cli, "discover_engine_backends", lambda _profile: {
        "backends": [{
            "id": "llama-cpp",
            "status": "ready",
            "models": ["qwen2.5:7b-instruct"],
        }],
    })

    saved = runner.invoke(engine_cli.engine_app, [
        "profile", "save", "Office local",
        "--backend", "llama-cpp",
        "--model", "qwen2.5:7b-instruct",
        "--json",
    ])
    assert saved.exit_code == 0, saved.stdout
    profile = json.loads(saved.stdout)["profile"]
    assert profile["backend_id"] == "llama-cpp"
    assert "api_key" not in saved.stdout and "endpoint" not in saved.stdout

    listed = runner.invoke(engine_cli.engine_app, ["profile", "list", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)["total"] == 1

    connected = runner.invoke(engine_cli.engine_app, ["profile", "connect", profile["id"], "--json"])
    assert connected.exit_code == 0, connected.stdout
    result = json.loads(connected.stdout)
    assert result["status"] == "verified"
    assert result["persistent_connection"] is False
    assert result["credentials_persisted"] is False

    saved_content = EngineFabricProfileStore().path.read_text(encoding="utf-8")
    assert "api_key" not in saved_content and "endpoint" not in saved_content

    deleted = runner.invoke(engine_cli.engine_app, ["profile", "delete", profile["id"], "--yes", "--json"])
    assert deleted.exit_code == 0, deleted.stdout
    assert json.loads(deleted.stdout)["deleted"]["id"] == profile["id"]
    assert json.loads(runner.invoke(engine_cli.engine_app, ["profile", "list", "--json"]).stdout)["total"] == 0


def test_engine_fabric_profile_cli_connect_fails_closed_for_offline_or_missing_models(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    profile = EngineFabricProfileStore().save(
        name="Office local",
        backend_id="llama-cpp",
        model="qwen2.5:7b-instruct",
    )
    monkeypatch.setattr(engine_cli, "discover_engine_backends", lambda _profile: {
        "backends": [{"id": "llama-cpp", "status": "setup_required", "models": []}],
    })
    offline = runner.invoke(engine_cli.engine_app, ["profile", "connect", profile["id"], "--json"])
    assert offline.exit_code != 0
    assert "not ready" in offline.output.lower()

    monkeypatch.setattr(engine_cli, "discover_engine_backends", lambda _profile: {
        "backends": [{"id": "llama-cpp", "status": "ready", "models": ["different-model"]}],
    })
    missing = runner.invoke(engine_cli.engine_app, ["profile", "connect", profile["id"], "--json"])
    assert missing.exit_code != 0
    assert "exact saved model id" in missing.output.lower()
