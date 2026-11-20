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
    assert "api_key" not in serialized
    assert "secret" not in serialized.lower()
    assert set(json.loads(serialized)) == {"schema_version", "profiles"}
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
