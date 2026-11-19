from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from axiom.api.server import create_app
from axiom.core.workflows import WorkflowStore


def _save_workflow(name: str) -> dict:
    record = {
        "id": name.lower().replace(" ", "-"),
        "name": name,
        "created_at": "2026-11-14T00:00:00Z",
        "updated_at": "2026-11-14T00:00:00Z",
        "nodes": [
            {"id": "dataset", "type": "dataset.profile"},
            {"id": "report", "type": "report.export"},
        ],
        "edges": [{"source": "dataset", "target": "report"}],
    }
    WorkflowStore().save(record)
    return WorkflowStore().get(record["id"]) or record


def test_workflow_api_uses_cursor_pagination_and_search(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        _save_workflow("Alpha audit")
        _save_workflow("Beta training")
        _save_workflow("Gamma audit")

        first = client.get("/api/workflows", params={"limit": 2})
        assert first.status_code == 200
        first_page = first.json()
        assert [item["name"] for item in first_page["workflows"]] == ["Gamma audit", "Beta training"]
        assert first_page["total"] == 3
        assert first_page["has_more"] is True
        assert first_page["next_cursor"] == first_page["workflows"][-1]["id"]

        second = client.get(
            "/api/workflows",
            params={"limit": 2, "after_workflow_id": first_page["next_cursor"]},
        )
        assert second.status_code == 200
        second_page = second.json()
        assert [item["name"] for item in second_page["workflows"]] == ["Alpha audit"]
        assert second_page["offset"] == 2
        assert second_page["has_more"] is False
        assert second_page["next_cursor"] is None

        filtered = client.get("/api/workflows", params={"q": "AUDIT", "limit": 1})
        assert filtered.status_code == 200
        assert filtered.json()["total"] == 2
        assert filtered.json()["filters"]["search"] == "audit"
        assert filtered.json()["workflows"][0]["name"] == "Gamma audit"

        invalid = client.get(
            "/api/workflows",
            params={"after_workflow_id": "missing-cursor"},
        )
        assert invalid.status_code == 400


def test_workflow_api_rejects_oversized_page_and_search(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        assert client.get("/api/workflows", params={"limit": 51}).status_code == 422
        assert client.get("/api/workflows", params={"q": "x" * 121}).status_code == 422
