from __future__ import annotations

import json
import re

import pytest

from axiom.core.agent_profiles import AgentProfileError, AgentProfileStore
from axiom.core.experiments import ExperimentStore
from axiom.core.resource_ids import (
    RESOURCE_ID_FORMAT,
    is_resource_id,
    legacy_resource_id,
    new_resource_id,
)
from axiom.evaluation.store import EvaluationStore
from axiom.models.registry import ModelRegistry
from axiom.core.workflows import WorkflowStore


def test_new_resource_ids_follow_canonical_uuid4_hex_contract():
    values = [new_resource_id() for _ in range(256)]
    assert len(values) == len(set(values))
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in values)
    assert all(is_resource_id(value) for value in values)
    assert RESOURCE_ID_FORMAT == "uuid4-hex-v1"
    assert is_resource_id("not-an-id") is False
    assert is_resource_id("A" * 32) is False
    assert is_resource_id(None) is False


def test_legacy_ids_are_deterministic_scoped_and_do_not_match_new_id_generation():
    first = legacy_resource_id("model", "Demo:org/demo")
    second = legacy_resource_id("model", "Demo:org/demo")
    other_kind = legacy_resource_id("dataset", "Demo:org/demo")
    assert first == second
    assert is_resource_id(first)
    assert other_kind != first
    with pytest.raises(ValueError):
        legacy_resource_id("../model", "Demo:org/demo")
    with pytest.raises(ValueError):
        legacy_resource_id("model", "")


def test_legacy_model_registry_ids_remain_stable_and_are_persisted_on_next_write(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    path = root / "models.json"
    path.write_text(json.dumps([{"name": "Demo", "source": "org/demo"}]), encoding="utf-8")
    store = ModelRegistry(root, create=False)

    expected = legacy_resource_id("model", "Demo:org/demo")
    first = store.get(expected)
    second = store.list()[0]
    assert first is not None
    assert first.id == expected == second.id

    # A read is non-mutating; the explicit write persists the same migrated ID.
    assert "id" not in json.loads(path.read_text(encoding="utf-8"))[0]
    store.update(expected, {"favorite": True})
    assert json.loads(path.read_text(encoding="utf-8"))[0]["id"] == expected


def test_unversioned_workflow_records_migrate_without_changing_ids(tmp_path):
    path = tmp_path / "workflows.json"
    path.write_text(json.dumps([{
        "id": "legacy-workflow-id",
        "name": "Legacy",
        "created_at": "2026-11-01T00:00:00+00:00",
        "updated_at": "2026-11-01T00:00:00+00:00",
        "nodes": [],
        "edges": [],
    }]), encoding="utf-8")
    store = WorkflowStore(tmp_path)

    read = store.list()[0]
    assert read["id"] == "legacy-workflow-id"
    assert read["schema_version"] == 2
    assert read["revision"] == 1
    assert read["versions"] == []
    assert "schema_version" not in json.loads(path.read_text(encoding="utf-8"))[0]

    updated = store.update(
        "legacy-workflow-id",
        name="Migrated workflow",
        nodes=[{"id": "step", "type": "dataset.profile"}],
        edges=[],
    )
    assert updated is not None
    assert updated["id"] == "legacy-workflow-id"
    stored = json.loads(path.read_text(encoding="utf-8"))[0]
    assert stored["id"] == "legacy-workflow-id"
    assert stored["schema_version"] == 2
    assert stored["revision"] == 2


def test_unversioned_evaluation_records_migrate_on_next_write_and_future_schema_fails_closed(tmp_path):
    path = tmp_path / "evaluations.json"
    path.write_text(json.dumps([{"id": "legacy-evaluation-id", "report": {}}]), encoding="utf-8")
    store = EvaluationStore(tmp_path)

    legacy = store.get("legacy-evaluation-id")
    assert legacy is not None
    assert legacy["schema_version"] == 1
    assert "schema_version" not in json.loads(path.read_text(encoding="utf-8"))[0]

    store.add({"schema_version": 1, "id": "new-evaluation-id", "report": {}})
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert {record["id"] for record in persisted} == {
        "legacy-evaluation-id", "new-evaluation-id"
    }
    assert all(record["schema_version"] == 1 for record in persisted)

    path.write_text(json.dumps([{"id": "future-id", "schema_version": 2}]), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Unsupported evaluation schema version"):
        store.list_page()
    assert path.read_bytes() == original


def test_agent_profile_store_rejects_unversioned_documents_without_rewriting(tmp_path):
    store = AgentProfileStore(tmp_path)
    store._check_paths(create=True)
    legacy_shape = {
        "profiles": [{
            "id": "default",
            "name": "default",
            "version": 1,
            "task_capabilities": [],
            "created_at": "",
            "updated_at": "",
        }]
    }
    store.path.write_text(json.dumps(legacy_shape), encoding="utf-8")
    original = store.path.read_bytes()
    with pytest.raises(AgentProfileError, match="Unsupported agent-profile schema"):
        store.load()
    assert store.path.read_bytes() == original
