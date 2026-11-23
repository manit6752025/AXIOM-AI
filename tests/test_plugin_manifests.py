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


def test_published_plugin_manifest_schema_validates_example_with_standard_draft_2020_12_validator():
    from pathlib import Path

    from jsonschema import Draft202012Validator, FormatChecker

    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "docs/schemas/plugin-manifest-v1.schema.json").read_text(encoding="utf-8"))
    example = json.loads((root / "examples/plugin-manifest.example.json").read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(example)
    validator.validate(manifest())

    invalid = {**example, "entrypoint": "module:run"}
    assert list(validator.iter_errors(invalid))


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
    store.register(manifest("1.0.0-alpha.10"))
    store.register(manifest("1.0.0"))
    with pytest.raises(PluginManifestError, match="newer"):
        store.register(manifest("1.0.0-beta.1"))
    assert len(store.list()["manifests"]) == 1


def test_manifest_semver_rejects_empty_segments_and_compares_numeric_prerelease_identifiers():
    for version in ("1.0.0-alpha..1", "1.0.0+build..2", "1.0.0-alpha.01"):
        with pytest.raises(PluginManifestError):
            validate_plugin_manifest(manifest(version))

    _, first_hash = validate_plugin_manifest(manifest("1.0.0-alpha.2"))
    _, second_hash = validate_plugin_manifest(manifest("1.0.0-alpha.10"))
    assert first_hash != second_hash


def test_manifest_store_serializes_concurrent_registrations_without_lost_updates(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    root = tmp_path / ".axiom"

    def register(index: int) -> str:
        candidate = manifest()
        candidate["id"] = f"org.example.plugin{index}"
        return PluginManifestStore(root).register(candidate)["manifest"]["id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        registered = list(executor.map(register, range(16)))

    assert len(set(registered)) == 16
    records = PluginManifestStore(root).list()["manifests"]
    assert {record["manifest"]["id"] for record in records} == set(registered)


def test_manifest_store_files_and_directory_use_private_permissions(tmp_path):
    import os
    import stat

    if os.name != "posix":
        pytest.skip("POSIX permission bits are not portable to this platform")
    store = PluginManifestStore(tmp_path / ".axiom")
    store.register(manifest())
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.registry_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.lock_file.stat().st_mode) == 0o600


def test_manifest_registry_lock_and_registry_symlinks_are_rejected_without_writes(tmp_path):
    store = PluginManifestStore(tmp_path / ".axiom")
    registered = store.register(manifest())
    lock_target = tmp_path / "outside-lock-target"
    lock_target.write_text("outside sentinel", encoding="utf-8")
    store.lock_file.unlink()
    try:
        store.lock_file.symlink_to(lock_target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PluginManifestError):
        store.grant(
            "com.example.research",
            expected_sha256=registered["sha256"],
            tools=[{"name": "search", "scopes": ["read"]}],
        )
    assert lock_target.read_text(encoding="utf-8") == "outside sentinel"

    store.lock_file.unlink()
    registry_target = tmp_path / "outside-registry-target"
    registry_target.write_text("outside registry sentinel", encoding="utf-8")
    store.registry_file.unlink()
    store.registry_file.symlink_to(registry_target)
    with pytest.raises(PluginManifestError):
        store.list()
    assert registry_target.read_text(encoding="utf-8") == "outside registry sentinel"


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


def test_manifest_store_rejects_malformed_unhashable_grant_scopes_and_duplicate_json_keys(tmp_path):
    store = PluginManifestStore(tmp_path / ".axiom")
    record = store.register(manifest())
    stored = json.loads(store.registry_file.read_text(encoding="utf-8"))
    stored["grants"]["com.example.research"] = {
        "manifest_sha256": record["sha256"],
        "tools": [{"name": "search", "scopes": [["read"]]}],
    }
    store.registry_file.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(PluginManifestError):
        store.list()

    store.registry_file.write_text(
        '{"store_schema_version":1,"store_schema_version":1,"manifests":[],"grants":{}}',
        encoding="utf-8",
    )
    with pytest.raises(PluginManifestError, match="safely"):
        store.list()


def test_empty_plugin_grants_are_rejected_and_leave_default_deny(tmp_path):
    store = PluginManifestStore(tmp_path / ".axiom")
    registered = store.register(manifest())
    with pytest.raises(PluginManifestError, match="at least one"):
        store.grant(
            "com.example.research",
            expected_sha256=registered["sha256"],
            tools=[],
        )
    assert store.list()["manifests"][0]["permission_status"] == "default_deny"
    assert store.is_allowed("com.example.research", "search", ["read"]) is False


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

        empty_grant = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={"manifest_sha256": digest, "tools": []},
        )
        assert empty_grant.status_code == 422

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


def test_plugin_manifest_api_reports_durable_audit_failure_after_registry_mutation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app
    from axiom.core.plugin_manifests import PluginManifestStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "axiom.api.security.record_security_event",
        lambda _event_type: (_ for _ in ()).throw(OSError("journal unavailable")),
    )
    with TestClient(create_app()) as client:
        response = client.post("/api/security/plugins/manifests", json={"manifest": manifest()})
        assert response.status_code == 503
        assert "may already have taken effect" in response.json()["detail"]
    assert PluginManifestStore().list()["manifests"][0]["manifest"]["id"] == "com.example.research"


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
        registered = client.post("/api/security/plugins/manifests", json={"manifest": manifest()})
        assert registered.status_code == 201
        digest = registered.json()["sha256"]
        plugin_id = registered.json()["manifest"]["id"]
        coerced_scope = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={"manifest_sha256": digest, "tools": [{"name": "search", "scopes": [1]}]},
        )
        assert coerced_scope.status_code == 422
        coerced_name = client.put(
            f"/api/security/plugins/{plugin_id}/permissions",
            json={"manifest_sha256": digest, "tools": [{"name": 7, "scopes": ["read"]}]},
        )
        assert coerced_name.status_code == 422

def test_plugin_manifest_cli_validate_register_grant_list_and_revoke(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from axiom.cli.main import app

    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "plugin.json"
    manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
    runner = CliRunner()

    validated = runner.invoke(app, [
        "--no-banner", "plugin", "manifest", "validate", str(manifest_path), "--json",
    ])
    assert validated.exit_code == 0, validated.output
    validation_report = json.loads(validated.stdout)
    assert validation_report["valid"] is True
    assert validation_report["permission_status"] == "default_deny"

    registered = runner.invoke(app, [
        "--no-banner", "plugin", "manifest", "register", str(manifest_path), "--json",
    ])
    assert registered.exit_code == 0, registered.output
    registration = json.loads(registered.stdout)
    digest = registration["sha256"]

    denied_listing = runner.invoke(app, ["--no-banner", "plugin", "list", "--json"])
    assert denied_listing.exit_code == 0, denied_listing.output
    assert json.loads(denied_listing.stdout)["manifests"][0]["permission_status"] == "default_deny"

    granted = runner.invoke(app, [
        "--no-banner", "plugin", "grant", "com.example.research",
        "--manifest-sha256", digest,
        "--tool", "search:read",
        "--tool", "annotate:read",
        "--json",
    ])
    assert granted.exit_code == 0, granted.output
    assert {item["name"] for item in json.loads(granted.stdout)["tools"]} == {"search", "annotate"}

    revoked = runner.invoke(app, ["--no-banner", "plugin", "revoke", "com.example.research"])
    assert revoked.exit_code == 0, revoked.output
    assert "Revoked" in revoked.stdout


def test_plugin_manifest_cli_rejects_duplicate_json_keys_and_symlink_files(tmp_path):
    from axiom.cli.plugins import _read_manifest

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"manifest_schema_version":1,"manifest_schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(PluginManifestError, match="duplicate"):
        _read_manifest(str(duplicate))

    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PluginManifestError, match="non-symlink"):
        _read_manifest(str(linked))

def test_published_manifest_schema_and_example_match_runtime_validator(tmp_path):
    from pathlib import Path

    from axiom.core.plugin_manifests import _PLUGIN_VERSION

    repository = Path(__file__).resolve().parents[1]
    schema = json.loads((repository / "docs/schemas/plugin-manifest-v1.schema.json").read_text(encoding="utf-8"))
    example = json.loads((repository / "examples/plugin-manifest.example.json").read_text(encoding="utf-8"))
    assert schema["properties"]["version"]["pattern"] == _PLUGIN_VERSION.pattern
    normalized, _digest = validate_plugin_manifest(example)
    assert normalized["id"] == "com.example.research"
    assert schema["properties"]["tools"]["maxItems"] == 32
