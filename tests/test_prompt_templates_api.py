from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import prompt_templates
from axiom.core.prompt_templates import PromptTemplateStore


def _client(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_templates, "PromptTemplateStore", lambda: PromptTemplateStore(tmp_path))
    app = FastAPI()
    app.include_router(prompt_templates.router)
    return TestClient(app)


def test_prompt_template_api_create_list_render_history_update_and_delete(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    created = client.post("/api/prompt-templates", json={
        "name": "Research brief",
        "description": "A short, structured brief.",
        "body": "Write a {{tone}} brief about {{topic}}.",
        "variables": [
            {"name": "tone", "type": "choice", "choices": ["formal", "friendly"]},
            {"name": "topic", "type": "string", "required": True, "min_length": 1, "max_length": 120},
        ],
    })
    assert created.status_code == 201, created.text
    template = created.json()["template"]
    template_id = template["id"]
    assert template["version"] == 1

    page = client.get("/api/prompt-templates?limit=10&offset=0&search=research")
    assert page.status_code == 200
    assert page.json()["total"] == 1
    assert page.json()["templates"][0]["id"] == template_id

    detail = client.get(f"/api/prompt-templates/{template_id}")
    assert detail.status_code == 200
    assert detail.json()["template"]["body"] == "Write a {{tone}} brief about {{topic}}."

    rendered = client.post(f"/api/prompt-templates/{template_id}/render", json={
        "values": {"tone": "friendly", "topic": "local inference"},
    })
    assert rendered.status_code == 200
    assert rendered.json()["rendered"] == "Write a friendly brief about local inference."
    assert rendered.json()["execution_performed"] is False
    assert rendered.json()["model_invoked"] is False

    wrong_type = client.post(f"/api/prompt-templates/{template_id}/render", json={
        "values": {"tone": "friendly", "topic": 42},
    })
    assert wrong_type.status_code == 422

    updated = client.put(f"/api/prompt-templates/{template_id}", json={
        "expected_version": 1,
        "name": "Research brief v2",
        "description": "Updated brief.",
        "body": "Prepare a {{tone}} brief about {{topic}}.",
        "variables": [
            {"name": "tone", "type": "choice", "choices": ["formal", "friendly"]},
            {"name": "topic", "type": "string", "required": True},
        ],
    })
    assert updated.status_code == 200, updated.text
    assert updated.json()["template"]["version"] == 2

    stale = client.put(f"/api/prompt-templates/{template_id}", json={
        "expected_version": 1, "name": "Stale", "body": "Stale", "variables": [],
    })
    assert stale.status_code == 409

    versions = client.get(f"/api/prompt-templates/{template_id}/versions")
    assert versions.status_code == 200
    assert [item["version"] for item in versions.json()["versions"]] == [2, 1]
    revision = client.get(f"/api/prompt-templates/{template_id}/versions/1")
    assert revision.status_code == 200
    assert revision.json()["revision"]["body"] == "Write a {{tone}} brief about {{topic}}."

    stale_delete = client.delete(f"/api/prompt-templates/{template_id}?expected_version=1")
    assert stale_delete.status_code == 409
    deleted = client.delete(f"/api/prompt-templates/{template_id}?expected_version=2")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(f"/api/prompt-templates/{template_id}").status_code == 404


def test_prompt_template_api_rejects_unbounded_or_unsafe_schema_inputs(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    invalid_type = client.post("/api/prompt-templates", json={
        "name": "Bad variable", "body": "{{value}}",
        "variables": [{"name": "value", "type": "python", "required": True}],
    })
    assert invalid_type.status_code == 422

    extra_field = client.post("/api/prompt-templates", json={
        "name": "Bad fields", "body": "Prompt", "variables": [],
        "run_command": "echo unsafe",
    })
    assert extra_field.status_code == 422

    oversized = client.post("/api/prompt-templates", json={
        "name": "Too many", "body": "Prompt",
        "variables": [{"name": f"value{index}", "type": "string"} for index in range(33)],
    })
    assert oversized.status_code == 422

    malformed_values = client.post("/api/prompt-templates/00000000000000000000000000000000/render", json={
        "values": {"bad": {"nested": "object"}},
    })
    assert malformed_values.status_code == 422
