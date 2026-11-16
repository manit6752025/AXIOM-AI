from __future__ import annotations

from axiom.core.workflows import WorkflowStore


def _record(workflow_id: str, name: str, node_type: str = "dataset.profile") -> dict:
    return {
        "id": workflow_id,
        "name": name,
        "created_at": "2026-11-14T00:00:00Z",
        "updated_at": "2026-11-14T00:00:00Z",
        "nodes": [{"id": "step", "type": node_type}],
        "edges": [],
    }


def test_workflow_store_pages_newest_first_with_stable_cursor(tmp_path) -> None:
    store = WorkflowStore(tmp_path / ".axiom")
    store.save(_record("first", "First audit"))
    store.save(_record("second", "Second training"))
    store.save(_record("third", "Third audit"))

    first = store.list_page(limit=2)
    assert [item["id"] for item in first["workflows"]] == ["third", "second"]
    assert first["total"] == 3
    assert first["offset"] == 0
    assert first["has_more"] is True
    assert first["next_cursor"] == "second"

    second = store.list_page(limit=2, after_workflow_id=first["next_cursor"])
    assert [item["id"] for item in second["workflows"]] == ["first"]
    assert second["offset"] == 2
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_workflow_store_search_applies_before_cursor_and_counts_filtered_items(tmp_path) -> None:
    store = WorkflowStore(tmp_path / ".axiom")
    store.save(_record("first", "First audit"))
    store.save(_record("second", "Second training", "training.plan"))
    store.save(_record("third", "Third audit"))

    page = store.list_page(limit=1, search="AUDIT")
    assert page["total"] == 2
    assert [item["name"] for item in page["workflows"]] == ["Third audit"]
    assert page["next_cursor"] == "third"

    next_page = store.list_page(limit=1, after_workflow_id=page["next_cursor"], search="audit")
    assert [item["name"] for item in next_page["workflows"]] == ["First audit"]
    assert next_page["has_more"] is False


def test_workflow_store_rejects_cursor_that_is_not_in_filtered_set(tmp_path) -> None:
    store = WorkflowStore(tmp_path / ".axiom")
    store.save(_record("first", "First audit"))
    store.save(_record("second", "Second training"))

    try:
        store.list_page(after_workflow_id="second", search="audit")
    except KeyError as exc:
        assert "selected filter" in str(exc)
    else:
        raise AssertionError("cursor from outside the filtered collection must be rejected")
