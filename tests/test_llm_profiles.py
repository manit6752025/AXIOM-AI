from __future__ import annotations

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from axiom.api import llm_client as llm_api
from axiom.cli.llm import llm_app
from axiom.core import llm_client
from axiom.core.llm_profiles import LLMProfileError, LLMProfileStore


def _api_client() -> TestClient:
    app = FastAPI()
    app.include_router(llm_api.router)
    return TestClient(app)


def test_profile_store_saves_only_non_secret_metadata_and_upserts_case_insensitively(tmp_path) -> None:
    store = LLMProfileStore(tmp_path)
    first = store.save(name="Team model", provider="deepseek", model="deepseek-chat")
    updated = store.save(name="team MODEL", provider="deepseek", model="deepseek-reasoner")

    assert first["id"] == updated["id"]
    assert updated["name"] == "team MODEL"
    assert updated["model"] == "deepseek-reasoner"
    report = store.list()
    assert report["total"] == 1
    assert report["credentials_persisted"] is False
    serialized = store.path.read_text(encoding="utf-8")
    persisted = json.loads(serialized)
    assert all("api_key" not in profile for profile in persisted["profiles"])
    assert all("secret" not in key.lower() for profile in persisted["profiles"] for key in profile)
    assert set(persisted) == {"schema_version", "profiles"}
    if os.name != "nt":
        assert (store.path.stat().st_mode & 0o777) & 0o077 == 0


def test_profile_store_validates_provider_url_limits_and_secret_fields(tmp_path) -> None:
    store = LLMProfileStore(tmp_path)
    with pytest.raises(LLMProfileError, match="supported AXIOM LLM provider"):
        store.save(name="Invalid", provider="unknown-provider", model="model")
    with pytest.raises(LLMProfileError, match="public HTTPS"):
        store.save(
            name="Private endpoint",
            provider="openai-compatible",
            model="model",
            base_url="http://127.0.0.1:8080/v1",
        )
    profile = store.save(
        name="Custom endpoint",
        provider="openai-compatible",
        model="model",
        base_url="https://93.184.216.34/api/v1/",
    )
    assert profile["base_url"] == "https://93.184.216.34/api/v1"
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["profiles"][0]["api_key"] = "must-not-be-persisted"
    store.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(LLMProfileError, match="secret-bearing fields"):
        store.list()


def test_profile_store_is_bounded_and_supports_explicit_deletion(tmp_path) -> None:
    store = LLMProfileStore(tmp_path)
    profile = store.save(name="Delete me", provider="axiom-local", model="local-model")
    assert store.get(profile["id"]) == profile
    assert store.delete(profile["id"]) == profile
    assert store.get(profile["id"]) is None
    with pytest.raises(LLMProfileError, match="Invalid provider-profile ID"):
        store.delete("../outside")


def test_profile_api_crud_and_catalogue_connect_never_persist_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_api, "LLMProfileStore", lambda: LLMProfileStore(tmp_path))
    captured = {}

    def fake_list_models(provider, api_key=None, base_url=None):
        captured.update(provider=provider, api_key=api_key, base_url=base_url)
        return {
            "provider": provider,
            "models": ["deepseek-chat"],
            "credential_storage": "not_saved",
            "execution": "remote",
        }

    monkeypatch.setattr(llm_client, "list_models", fake_list_models)
    client = _api_client()
    created = client.post("/api/llm/profiles", json={
        "name": "DeepSeek team",
        "provider": "deepseek",
        "model": "deepseek-chat",
    })
    assert created.status_code == 201, created.text
    profile = created.json()["profile"]
    assert created.json()["credentials_persisted"] is False

    listed = client.get("/api/llm/profiles")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["profiles"][0]["id"] == profile["id"]

    connected = client.post(f"/api/llm/profiles/{profile['id']}/connect", json={"api_key": "request-only-sentinel"})
    assert connected.status_code == 200, connected.text
    assert connected.json()["status"] == "verified"
    assert connected.json()["persistent_connection"] is False
    assert connected.json()["credential_storage"] == "not_saved"
    assert captured == {"provider": "deepseek", "api_key": "request-only-sentinel", "base_url": None}
    assert "request-only-sentinel" not in connected.text
    assert "request-only-sentinel" not in LLMProfileStore(tmp_path).path.read_text(encoding="utf-8")

    bad = client.post("/api/llm/profiles", json={
        "name": "Has secret",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key": "never-accept-profile-secrets",
    })
    assert bad.status_code == 422
    deleted = client.delete(f"/api/llm/profiles/{profile['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"]["id"] == profile["id"]
    assert client.get("/api/llm/profiles").json()["total"] == 0


def test_profile_cli_saves_lists_connects_and_deletes_without_persisting_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIOM_LLM_API_KEY", "cli-request-only-sentinel")
    captured = {}

    def fake_list_models(provider, api_key=None, base_url=None):
        captured.update(provider=provider, api_key=api_key, base_url=base_url)
        return {"models": ["deepseek-chat"], "credential_storage": "not_saved"}

    monkeypatch.setattr(llm_client, "list_models", fake_list_models)
    runner = CliRunner()
    saved = runner.invoke(llm_app, [
        "profile", "save", "CLI team",
        "--provider", "deepseek",
        "--model", "deepseek-chat",
        "--json",
    ])
    assert saved.exit_code == 0, saved.stdout
    profile = json.loads(saved.stdout)["profile"]
    assert "cli-request-only-sentinel" not in saved.stdout

    listed = runner.invoke(llm_app, ["profile", "list", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)["total"] == 1

    connected = runner.invoke(llm_app, ["profile", "connect", profile["id"], "--json"])
    assert connected.exit_code == 0, connected.stdout
    assert json.loads(connected.stdout)["status"] == "verified"
    assert captured["api_key"] == "cli-request-only-sentinel"
    assert "cli-request-only-sentinel" not in LLMProfileStore().path.read_text(encoding="utf-8")

    deleted = runner.invoke(llm_app, ["profile", "delete", profile["id"], "--yes", "--json"])
    assert deleted.exit_code == 0, deleted.stdout
    assert json.loads(deleted.stdout)["deleted"]["id"] == profile["id"]



def test_profile_store_schema_v1_reads_and_migrates_on_next_write(tmp_path) -> None:
    legacy = {
        "schema_version": 1,
        "profiles": [{
            "id": "a" * 32,
            "name": "Legacy team",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }],
    }
    path = tmp_path / "llm-provider-profiles.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    store = LLMProfileStore(tmp_path)

    report = store.list()
    assert report["schema_version"] == 2
    assert report["profiles"][0]["api_key_env"] is None
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1

    store.save(name="New reference", provider="deepseek", model="deepseek-chat", api_key_env="TEAM_DEEPSEEK_KEY")
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 2
    assert migrated["profiles"][0]["api_key_env"] == "TEAM_DEEPSEEK_KEY"
    assert all("api_key" not in item for item in migrated["profiles"])
    assert "TEAM_DEEPSEEK_KEY" in path.read_text(encoding="utf-8")


def test_profile_store_validates_environment_reference_names_and_provider_scope(tmp_path) -> None:
    store = LLMProfileStore(tmp_path)
    for invalid in ("bad-name", "with space", "TOKEN\nINJECTED", "$TOKEN", "A" * 129, ""):
        if invalid == "":
            continue
        with pytest.raises(LLMProfileError, match="environment reference"):
            store.save(name="Invalid", provider="deepseek", model="deepseek-chat", api_key_env=invalid)
    with pytest.raises(LLMProfileError, match="Local providers"):
        store.save(name="Local", provider="axiom-local", model="tiny", api_key_env="LOCAL_KEY")


def test_profile_api_resolves_environment_reference_only_inside_provider_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_api, "LLMProfileStore", lambda: LLMProfileStore(tmp_path))
    monkeypatch.setenv("AXIOM_TEST_DEEPSEEK_KEY", "profile-secret-sentinel")
    captured = {}

    def fake_list_models(provider, api_key=None, base_url=None):
        captured.update(provider=provider, api_key=api_key, base_url=base_url)
        return {
            "provider": provider,
            "models": ["deepseek-chat"],
            "credential_storage": "not_saved",
            "execution": "remote",
        }

    monkeypatch.setattr(llm_client, "list_models", fake_list_models)
    client = _api_client()
    created = client.post("/api/llm/profiles", json={
        "name": "Server credential",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "AXIOM_TEST_DEEPSEEK_KEY",
    })
    assert created.status_code == 201, created.text
    profile = created.json()["profile"]
    assert profile["api_key_env"] == "AXIOM_TEST_DEEPSEEK_KEY"
    assert "profile-secret-sentinel" not in created.text

    connected = client.post(f"/api/llm/profiles/{profile['id']}/connect", json={})
    assert connected.status_code == 200, connected.text
    assert connected.json()["credential_source"] == "environment_reference"
    assert captured == {
        "provider": "deepseek",
        "api_key": "profile-secret-sentinel",
        "base_url": None,
    }
    assert "profile-secret-sentinel" not in connected.text
    assert "profile-secret-sentinel" not in LLMProfileStore(tmp_path).path.read_text(encoding="utf-8")

    monkeypatch.delenv("AXIOM_TEST_DEEPSEEK_KEY")
    missing = client.post(f"/api/llm/profiles/{profile['id']}/connect", json={})
    assert missing.status_code == 422
    assert "profile-secret-sentinel" not in missing.text


def test_profile_cli_uses_only_the_saved_environment_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEAM_DEEPSEEK_KEY", "cli-profile-secret-sentinel")
    captured = {}
    monkeypatch.setattr(llm_client, "list_models", lambda provider, api_key=None, base_url=None: (
        captured.update(provider=provider, api_key=api_key, base_url=base_url)
        or {"models": ["deepseek-chat"], "credential_storage": "not_saved"}
    ))
    runner = CliRunner()
    saved = runner.invoke(llm_app, [
        "profile", "save", "Environment team",
        "--provider", "deepseek",
        "--model", "deepseek-chat",
        "--api-key-env", "TEAM_DEEPSEEK_KEY",
        "--json",
    ])
    assert saved.exit_code == 0, saved.stdout
    saved_profile = json.loads(saved.stdout)["profile"]
    assert saved_profile["api_key_env"] == "TEAM_DEEPSEEK_KEY"
    assert "cli-profile-secret-sentinel" not in saved.stdout

    connected = runner.invoke(llm_app, ["profile", "connect", saved_profile["id"], "--json"])
    assert connected.exit_code == 0, connected.stdout
    assert captured["api_key"] == "cli-profile-secret-sentinel"
    assert "cli-profile-secret-sentinel" not in connected.stdout
    assert "cli-profile-secret-sentinel" not in LLMProfileStore().path.read_text(encoding="utf-8")
