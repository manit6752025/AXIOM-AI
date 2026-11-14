from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import workspace


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(workspace.router)
    return TestClient(app)


def test_dataset_inventory_paginates_in_stable_path_order(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("zeta.jsonl", "alpha.jsonl", "middle.jsonl", "notes.txt"):
        (data / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    client = _client()
    first = client.get("/api/workspace/datasets", params={"limit": 2, "offset": 0})
    second = client.get("/api/workspace/datasets", params={"limit": 2, "offset": 2})

    assert first.status_code == second.status_code == 200
    assert [item["path"] for item in first.json()["datasets"]] == ["alpha.jsonl", "middle.jsonl"]
    assert [item["path"] for item in second.json()["datasets"]] == ["zeta.jsonl"]
    assert first.json()["total"] == second.json()["total"] == 3
    assert first.json()["has_more"] is True
    assert second.json()["has_more"] is False
    assert first.json()["truncated"] is False
    assert first.json()["scan_limit"] == 5000


def test_dataset_inventory_search_is_case_insensitive_and_applied_before_paging(
    tmp_path: Path, monkeypatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train-a.jsonl", "train-b.jsonl", "valid.jsonl"):
        (data / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    response = _client().get(
        "/api/workspace/datasets",
        params={"search": "TRAIN", "limit": 1, "offset": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["path"] for item in payload["datasets"]] == ["train-b.jsonl"]
    assert payload["total"] == 2
    assert payload["has_more"] is False
    assert payload["filters"]["search"] == "train"


def test_dataset_inventory_discloses_bounded_scan_truncation(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for index in range(5001):
        (data / f"dataset-{index:04}.jsonl").touch()
    monkeypatch.chdir(tmp_path)

    response = _client().get("/api/workspace/datasets", params={"limit": 500})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["datasets"]) == 500
    assert payload["total"] == 5000
    assert payload["truncated"] is True
    assert payload["scan_limit"] == 5000


def test_workspace_events_are_paginated_newest_first_and_searchable() -> None:
    with workspace._events_lock:
        workspace._events.clear()
    workspace.record_workspace_event("models.add", "Added Alpha")
    workspace.record_workspace_event("datasets.profile", "Profiled train.jsonl")
    workspace.record_workspace_event("models.remove", "Removed Beta")
    client = _client()

    first = client.get("/api/workspace/events", params={"limit": 2, "offset": 0})
    second = client.get("/api/workspace/events", params={"limit": 2, "offset": 2})
    filtered = client.get("/api/workspace/events", params={"search": "MODELS"})

    assert first.status_code == second.status_code == filtered.status_code == 200
    assert [event["action"] for event in first.json()["events"]] == [
        "models.remove", "datasets.profile"
    ]
    assert [event["action"] for event in second.json()["events"]] == ["models.add"]
    assert first.json()["total"] == 3
    assert first.json()["has_more"] is True
    assert second.json()["has_more"] is False
    assert [event["action"] for event in filtered.json()["events"]] == [
        "models.remove", "models.add"
    ]
    assert filtered.json()["filters"]["search"] == "models"


def test_workspace_collection_pagination_rejects_out_of_bounds_limits() -> None:
    client = _client()
    assert client.get("/api/workspace/datasets", params={"limit": 501}).status_code == 422
    assert client.get("/api/workspace/events", params={"limit": 101}).status_code == 422
