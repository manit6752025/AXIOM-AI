from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from axiom.core import security_events
from axiom.core.security_events import SecurityEventError, SecurityEventStore


def test_security_event_store_records_allowlisted_events_and_verifies_chain(tmp_path: Path):
    store = SecurityEventStore(tmp_path)

    first = store.record("mcp.profile.created")
    second = store.record("mcp.permissions.saved")

    report = store.verify()
    page = store.list_page()

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert first["prev_hash"] == "0" * 64
    assert second["prev_hash"] == first["hash"]
    assert report["valid"] is True
    assert report["event_count"] == 2
    assert report["head_hash"] == second["hash"]
    assert [event["event_type"] for event in page["events"]] == [
        "mcp.permissions.saved",
        "mcp.profile.created",
    ]
    assert page["integrity"]["valid"] is True


def test_security_event_store_rejects_unrecognized_event_text(tmp_path: Path):
    store = SecurityEventStore(tmp_path)

    with pytest.raises(SecurityEventError, match="not allowlisted"):
        store.record("mcp.profile.created SECRET_VALUE")

    with pytest.raises(SecurityEventError, match="not allowlisted"):
        store.record("custom event with user-provided summary")


def test_security_event_review_appends_acknowledgement_without_rewriting_original(tmp_path: Path):
    store = SecurityEventStore(tmp_path)
    original = store.record("mcp.permissions.saved")

    result = store.review(original["event_id"], "Security Admin")
    repeated = store.review(original["event_id"], "Another Admin")
    page = store.list_page()
    export = store.export()

    reviewed = page["events"][0]
    assert result["already_reviewed"] is False
    assert repeated["already_reviewed"] is True
    assert result["review"]["reviewer_label"] == "Security Admin"
    assert result["review"]["target_event_id"] == original["event_id"]
    assert reviewed["event_id"] == original["event_id"]
    assert reviewed["reviewed"] is True
    assert reviewed["reviewer_label"] == "Security Admin"
    assert reviewed["reviewer_identity_verified"] is False
    assert export["integrity"]["valid"] is True
    assert len(export["events"]) == 2
    assert export["events"][0]["hash"] == original["hash"]


def test_security_event_review_requires_safe_self_attested_label(tmp_path: Path):
    store = SecurityEventStore(tmp_path)
    event = store.record("mcp.profile.activated")

    with pytest.raises(SecurityEventError, match="2–48"):
        store.review(event["event_id"], "x")
    with pytest.raises(SecurityEventError, match="safe characters"):
        store.review(event["event_id"], "Admin\nInjected")
    with pytest.raises(SecurityEventError, match="not found"):
        store.review("0" * 32, "Admin User")


def test_security_event_store_detects_modified_or_removed_records(tmp_path: Path):
    store = SecurityEventStore(tmp_path)
    store.record("mcp.profile.created")
    path = tmp_path / ".axiom" / "security-events.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["events"][0]["summary"] = "Modified by attacker"
    path.write_text(json.dumps(document), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)

    with pytest.raises(SecurityEventError, match="hash chain verification failed|not allowlisted"):
        store.verify()


def test_security_event_store_refuses_silent_rotation_when_capacity_is_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(security_events, "MAX_EVENTS", 2)
    store = SecurityEventStore(tmp_path)
    first = store.record("mcp.profile.created")
    second = store.record("mcp.permissions.saved")

    with pytest.raises(SecurityEventError, match="journal is full"):
        store.record("mcp.profile.activated")

    exported = store.export()
    assert exported["integrity"]["capacity_reached"] is True
    assert [event["event_id"] for event in exported["events"]] == [first["event_id"], second["event_id"]]


def test_security_event_store_serializes_concurrent_appenders(tmp_path: Path):
    store = SecurityEventStore(tmp_path)
    event_types = [
        "mcp.profile.created",
        "mcp.permissions.saved",
        "mcp.profile.activated",
    ] * 8

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store.record, event_types))

    report = store.verify()
    assert report["event_count"] == len(event_types)
    assert report["valid"] is True


def test_security_event_store_rejects_symlinked_journal_and_directory(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "security-events.json").write_text("{}", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".axiom").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SecurityEventError, match="symbolic link"):
        SecurityEventStore(workspace).record("mcp.profile.created")


def test_security_event_export_contains_no_credentials_or_arbitrary_text(tmp_path: Path):
    store = SecurityEventStore(tmp_path)
    store.record("mcp.profile.created")
    exported = json.dumps(store.export(), sort_keys=True)

    assert "api_token" not in exported.lower()
    assert "password" not in exported.lower()
    assert "prompt" not in exported.lower()
    assert "command_line" not in exported.lower()
    assert store.list_page()["privacy"]["credential_values_stored"] is False
    assert store.list_page()["privacy"]["reviewer_identity_authenticated"] is False


def test_security_event_read_operations_do_not_create_workspace_metadata(tmp_path: Path):
    store = SecurityEventStore(tmp_path)

    page = store.list_page()
    exported = store.export()
    verified = store.verify()

    assert page["events"] == []
    assert exported["events"] == []
    assert verified["valid"] is True
    assert not (tmp_path / ".axiom").exists()


def test_security_event_pagination_returns_newest_first_without_skipping_reviews(tmp_path: Path):
    store = SecurityEventStore(tmp_path)
    first = store.record("mcp.profile.created")
    second = store.record("mcp.permissions.saved")
    third = store.record("mcp.profile.activated")
    store.review(second["event_id"], "Reviewer")

    first_page = store.list_page(limit=2, offset=0)
    second_page = store.list_page(limit=2, offset=2)

    assert [event["event_id"] for event in first_page["events"]] == [
        third["event_id"],
        second["event_id"],
    ]
    assert first_page["events"][1]["reviewed"] is True
    assert [event["event_id"] for event in second_page["events"]] == [first["event_id"]]
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False


def test_security_event_store_rejects_symlinked_lock_file(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = workspace / ".axiom"
    metadata.mkdir()
    target = tmp_path / "outside-lock"
    target.write_text("preserve", encoding="utf-8")
    lock = metadata / "security-events.lock"
    try:
        lock.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks are unavailable on this platform: {exc}")

    with pytest.raises(SecurityEventError, match="symbolic link|lock"):
        SecurityEventStore(workspace).record("mcp.profile.created")
    assert target.read_text(encoding="utf-8") == "preserve"
