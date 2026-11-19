from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import workspace
from axiom.models.registry import Model


def _models() -> list[Model]:
    return [
        Model(name="Zeta", source="/models/zeta", format="gguf", parameters="7B", tags=["local"], favorite=True),
        Model(name="Alpha", source="/models/alpha", format="safetensors", parameters="13B", tags=["training"], favorite=False),
        Model(name="Qwen Mini", source="/models/qwen-mini", format="gguf", parameters="1.5B", tags=["local", "chat"], favorite=False),
        Model(name="Qwen Seven", source="/models/qwen-seven", format="gguf", parameters="7B", tags=["local", "chat"], favorite=True),
        Model(name="Unknown size", source="/models/unknown", format="gguf", parameters=None, tags=["local"], favorite=False),
    ]


def _client(monkeypatch) -> TestClient:
    records = _models()

    class StubRegistry:
        def list(self) -> list[Model]:
            return records

    monkeypatch.setattr(workspace, "ModelRegistry", StubRegistry)
    app = FastAPI()
    app.include_router(workspace.router)
    return TestClient(app)


def test_model_registry_search_filters_format_favorite_and_exact_tag(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get(
        "/api/workspace/models",
        params={
            "search": "QWEN",
            "favorite_only": "true",
            "format": "gguf",
            "tag": "chat",
            "sort_by": "name",
            "sort_order": "asc",
            "limit": "25",
            "offset": "0",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["name"] for item in payload["models"]] == ["Qwen Seven"]
    assert payload["total"] == 1
    assert payload["has_more"] is False
    assert payload["filters"]["search"] == "qwen"
    assert payload["filters"]["favorite_only"] is True


def test_model_registry_pagination_operates_after_filter_and_sort(monkeypatch) -> None:
    client = _client(monkeypatch)
    first = client.get("/api/workspace/models", params={"limit": 2, "offset": 0, "sort_by": "name"})
    second = client.get("/api/workspace/models", params={"limit": 2, "offset": 2, "sort_by": "name"})

    assert first.status_code == second.status_code == 200
    assert [item["name"] for item in first.json()["models"]] == ["Alpha", "Qwen Mini"]
    assert [item["name"] for item in second.json()["models"]] == ["Qwen Seven", "Unknown size"]
    assert first.json()["total"] == second.json()["total"] == 5
    assert first.json()["has_more"] is True
    assert second.json()["has_more"] is True


def test_model_registry_parameter_sort_is_numeric_and_unknown_sizes_stay_last(monkeypatch) -> None:
    client = _client(monkeypatch)
    ascending = client.get("/api/workspace/models", params={"sort_by": "parameters", "sort_order": "asc"})
    descending = client.get("/api/workspace/models", params={"sort_by": "parameters", "sort_order": "desc"})

    assert ascending.status_code == descending.status_code == 200
    assert [item["name"] for item in ascending.json()["models"]] == [
        "Qwen Mini", "Qwen Seven", "Zeta", "Alpha", "Unknown size"
    ]
    assert [item["name"] for item in descending.json()["models"]] == [
        "Alpha", "Qwen Seven", "Zeta", "Qwen Mini", "Unknown size"
    ]


def test_model_registry_rejects_unknown_sort_fields(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/api/workspace/models", params={"sort_by": "arbitrary-expression"})

    assert response.status_code == 422
