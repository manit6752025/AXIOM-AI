from __future__ import annotations

import json

import pytest

from axiom.core.plugin_manifests import (
    MAX_MANIFESTS,
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
    linked = root / "plugins-link"
    try:
        linked.symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")


def test_manifest_store_enforces_workspace_registry_capacity(tmp_path, monkeypatch):
    from axiom.core import plugin_manifests

    store = PluginManifestStore(tmp_path / ".axiom")
    monkeypatch.setattr(plugin_manifests, "MAX_MANIFESTS", 1)
    store.register(manifest())
    other = manifest()
    other["id"] = "org.example.second"
    with pytest.raises(PluginManifestError, match="at most"):
        store.register(other)
