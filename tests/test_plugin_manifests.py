from __future__ import annotations

import json

import pytest

from axiom.core.plugin_manifests import (
    PluginManifestError,
    PluginManifestStore,
    validate_plugin_manifest,
)


def manifest(version: str = "1.0.0", *, description: str = "A safe declarative plugin") -> dict:
    return {
        "manifest_schema_version": 1,
        "id": "com.example.research",
        "name": "Example research",
        "version": version,
        "description": description,
        "tools": [
            {
                "name": "search",
                "description": "Search a bounded local index.",
                "scopes": ["read"],
            },
            {
                "name": "annotate",
                "description": "Write an annotation after user approval.",
                "scopes": ["read", "write"],
            },
        ],
    }


def test_manifest_is_canonical_strict_and_disallows_executable_fields():
    normalized, digest = validate_plugin_manifest(manifest())
    assert normalized["id"] == "com.example.research"
    assert [item["name"] for item in normalized["tools"]] == ["annotate", "search"]
    assert len(digest) == 64
    assert digest == validate_plugin_manifest(manifest())[1]

    for invalid in (
        {**manifest(), "command": "sh -c dangerous"},
        {**manifest(), "entrypoint": "plugin:run"},
        {**manifest(), "tools": [{**manifest()["tools"][0], "scopes": ["read", "admin"]}]},
        {**manifest(), "tools": [{**manifest()["tools"][0], "name": "UpperCase"}]},
        {**manifest(), "manifest_schema_version": True},
        {**manifest(), "unknown": "value"},
    ):
        with pytest.raises(PluginManifestError):
            validate_plugin_manifest(invalid)


def test_manifest_grants_are_default_deny_scope_limited_and_hash_bound(tmp_path):
    store = PluginManifestStore(tmp_path / ".axiom")
    registered = store.register(manifest())
    digest = registered["sha256"]
    assert store.is_allowed("com.example.research", "search", ["read"]) is False
    assert store.list()["manifests"][0]["permission_status"] == "default_deny"

    with pytest.raises(PluginManifestError, match="subset"):
        store.grant(
            "com.example.research",
            expected_sha256=digest,
            tools=[{"name": "search", "scopes": ["write"]}],
        )

    store.grant(
        "com.example.research",
        expected_sha256=digest,
        tools=[
            {"name": "search", "scopes": ["read"]},
            {"name": "annotate", "scopes": ["read"]},
        ],
    )
    assert store.is_allowed("com.example.research", "search", ["read"]) is True
    assert store.is_allowed("com.example.research", "search", ["write"]) is False
    assert store.is_allowed("com.example.research", "annotate", ["read", "write"]) is False
    assert store.list()["manifests"][0]["permission_status"] == "granted"

    updated = store.register(manifest("1.1.0", description="A changed manifest"))
    assert updated["sha256"] != digest
    assert store.is_allowed("com.example.research", "search", ["read"]) is False
    assert store.list()["manifests"][0]["permission_status"] == "stale_manifest"

    with pytest.raises(PluginManifestError, match="newer"):
        store.register(manifest("1.0.1"))

    with pytest.raises(PluginManifestError, match="changed"):
        store.grant(
            "com.example.research",
            expected_sha256=digest,
            tools=[{"name": "search", "scopes": ["read"]}],
        )

    store.grant(
        "com.example.research",
        expected_sha256=updated["sha256"],
        tools=[{"name": "search", "scopes": ["read"]}],
    )
    assert store.is_allowed("com.example.research", "search", ["read"]) is True
    assert store.revoke("com.example.research") is True
    assert store.is_allowed("com.example.research", "search", ["read"]) is False
    assert store.revoke("com.example.research") is False


def test_manifest_versions_follow_semver_and_duplicate_ids_are_rejected(tmp_path):
    store = PluginManifestStore(tmp_path / ".axiom")
    store.register(manifest("1.0.0-alpha.1"))
    store.register(manifest("1.0.0-alpha.2"))
    store.register(manifest("1.0.0"))
    with pytest.raises(PluginManifestError, match="newer"):
        store.register(manifest("1.0.0-beta.1"))
    assert len(store.list()["manifests"]) == 1


def test_manifest_store_fails_closed_for_tampered_registry_and_symlinks(tmp_path):
    root = tmp_path / ".axiom"
    store = PluginManifestStore(root)
    store.register(manifest())
    raw = json.loads(store.registry_file.read_text(encoding="utf-8"))
    raw["manifests"][0]["manifest"]["tools"][0]["scopes"] = ["admin"]
    store.registry_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PluginManifestError):
        store.list()

    other = tmp_path / "external"
    other.mkdir()
    linked_metadata = tmp_path / "linked-metadata"
    try:
        linked_metadata.symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PluginManifestError, match="symbolic links"):
        PluginManifestStore(linked_metadata).list()
    assert store.is_allowed("com.example.research", "search", []) is False


def test_manifest_store_enforces_workspace_registry_capacity(tmp_path, monkeypatch):
    from axiom.core import plugin_manifests

    store = PluginManifestStore(tmp_path / ".axiom")
    monkeypatch.setattr(plugin_manifests, "MAX_MANIFESTS", 1)
    store.register(manifest())
    other = manifest()
    other["id"] = "org.example.second"
    with pytest.raises(PluginManifestError, match="at most"):
        store.register(other)

def test_plugin_manifest_api_keeps_registration_default_deny_and_invalidates_grants(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        initial = client.get("/api/security/plugins")
        assert initial.status_code == 200
        assert initial.json()["manifests"] == []
        assert "does not load plugin code" in initial.json()["execution_note"]

        registered = client.post("/api/security/plugins/manifests", json={"manifest": manifest()})
        assert registered.status_code == 201, registered.text
        digest = registered.json()["sha256"]
        plugin_id = registered.json()["manifest"]["id"]
        inventory = client.get("/api/security/plugins").json()
        assert inventory["manifests"][0]["permission_status"] == "default_deny"

        denied = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={
                "manifest_sha256": digest,
                "tools": [{"name": "search", "scopes": ["write"]}],
            },
        )
        assert denied.status_code == 422

        grant = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={
                "manifest_sha256": digest,
                "tools": [{"name": "search", "scopes": ["read"]}],
            },
        )
        assert grant.status_code == 200, grant.text
        assert client.get("/api/security/plugins").json()["manifests"][0]["permission_status"] == "granted"

        wrong_type = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={"manifest_sha256": digest, "tools": [{"name": "search", "scopes": ["read"], "extra": True}]},
        )
        assert wrong_type.status_code == 422

        changed_manifest = manifest("1.1.0")
        changed = client.post("/api/security/plugins/manifests", json={"manifest": changed_manifest})
        assert changed.status_code == 201, changed.text
        assert changed.json()["sha256"] != digest
        assert client.get("/api/security/plugins").json()["manifests"][0]["permission_status"] == "stale_manifest"

        stale_grant = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={
                "manifest_sha256": digest,
                "tools": [{"name": "search", "scopes": ["read"]}],
            },
        )
        assert stale_grant.status_code == 422

        current_grant = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={
                "manifest_sha256": changed.json()["sha256"],
                "tools": [{"name": "search", "scopes": ["read"]}],
            },
        )
        assert current_grant.status_code == 200
        revoked = client.delete(f"/api/security/plugins/{plugin_id}/permissions")
        assert revoked.status_code == 200
        assert revoked.json()["revoked"] is True
        assert client.get("/api/security/plugins").json()["manifests"][0]["permission_status"] == "default_deny"

        events = client.get("/api/security/events/export").json()["events"]
        plugin_events = [event["event_type"] for event in events if event["event_type"].startswith("plugin.")]
        assert plugin_events == [
            "plugin.manifest.registered",
            "plugin.permissions.granted",
            "plugin.manifest.registered",
            "plugin.permissions.granted",
            "plugin.permissions.revoked",
        ]


def test_plugin_manifest_api_rejects_non_object_and_extra_envelope_fields(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        assert client.post("/api/security/plugins/manifests", json={"manifest": []}).status_code == 422
        assert client.post(
            "/api/security/plugins/manifests",
            json={"manifest": manifest(), "execute": True},
        ).status_code == 422
        assert client.delete("/api/security/plugins/INVALID/permissions").status_code == 422
