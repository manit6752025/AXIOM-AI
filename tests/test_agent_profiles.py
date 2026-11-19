from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.core.agent_profiles import AgentProfileError, AgentProfileStore, TASK_CAPABILITIES


def test_missing_store_returns_default_deny_without_writing(tmp_path: Path) -> None:
    store = AgentProfileStore(tmp_path)
    report = store.load()
    assert report["schema_version"] == 1
    assert len(report["profiles"]) == 1
    assert report["profiles"][0]["name"] == "default"
    assert report["profiles"][0]["task_capabilities"] == []
    assert not store.path.exists()


def test_create_update_delete_profile_and_versioning(tmp_path: Path) -> None:
    store = AgentProfileStore(tmp_path)
    profile = store.create("Review team", ["dataset.profile"])
    assert profile["version"] == 1
    updated = store.update_capabilities(profile["id"], ["model.inspect", "dataset.profile"])
    assert updated["version"] == 2
    assert updated["task_capabilities"] == ["dataset.profile", "model.inspect"]
    assert store.load()["profiles"][-1]["version"] == 2
    deleted = store.delete(profile["id"])
    assert deleted["deleted_id"] == profile["id"]
    assert [item["id"] for item in store.load()["profiles"]] == ["default"]


def test_duplicate_names_unknown_capability_and_default_delete_are_rejected(tmp_path: Path) -> None:
    store = AgentProfileStore(tmp_path)
    store.create("Research")
    with pytest.raises(AgentProfileError, match="already exists"):
        store.create("research")
    with pytest.raises(AgentProfileError, match="not in the supported catalog"):
        store.create("Unsupported", ["shell.execute"])
    with pytest.raises(AgentProfileError, match="default profile cannot be deleted"):
        store.delete("default")


def test_store_rejects_future_or_malformed_schema(tmp_path: Path) -> None:
    store = AgentProfileStore(tmp_path)
    store.directory.mkdir()
    store.path.write_text(json.dumps({"schema_version": 2, "profiles": []}), encoding="utf-8")
    with pytest.raises(AgentProfileError, match="Unsupported agent-profile schema"):
        store.load()
    store.path.write_text(json.dumps({"schema_version": True, "profiles": []}), encoding="utf-8")
    with pytest.raises(AgentProfileError, match="Unsupported agent-profile schema"):
        store.load()


def test_store_rejects_symlinked_metadata_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".axiom").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AgentProfileError, match="must not be a symbolic link"):
        AgentProfileStore(tmp_path).load()


def test_catalog_is_a_small_explicit_allowlist() -> None:
    assert TASK_CAPABILITIES
    assert len(TASK_CAPABILITIES) == len(set(TASK_CAPABILITIES))
    assert "shell.execute" not in TASK_CAPABILITIES


def test_store_fails_closed_when_permissions_cannot_be_restricted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axiom.core import agent_profiles

    store = AgentProfileStore(tmp_path)

    def fail_chmod(*args, **kwargs):
        raise OSError("permission change denied")

    monkeypatch.setattr(agent_profiles.os, "chmod", fail_chmod)

    with pytest.raises(
        AgentProfileError,
        match="Could not secure agent-profile file permissions",
    ):
        store.create("Private profile")

    assert not store.path.exists()
    assert not list(store.directory.glob(".agent-profiles-*.tmp"))
    