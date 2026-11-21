from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from axiom.core.prompt_templates import (
    PromptTemplateConflict,
    PromptTemplateError,
    PromptTemplateStore,
    render_prompt,
)


def _variables():
    return [
        {"name": "name", "type": "string", "required": True, "default": None, "description": "Person name",
         "choices": [], "min_length": 1, "max_length": 80, "minimum": None, "maximum": None},
        {"name": "count", "type": "integer", "required": False, "default": 3, "description": "",
         "choices": [], "min_length": None, "max_length": None, "minimum": 1, "maximum": 20},
        {"name": "temperature", "type": "number", "required": False, "default": 0.2, "description": "",
         "choices": [], "min_length": None, "max_length": None, "minimum": 0, "maximum": 2},
        {"name": "include_sources", "type": "boolean", "required": False, "default": True, "description": "",
         "choices": [], "min_length": None, "max_length": None, "minimum": None, "maximum": None},
        {"name": "tone", "type": "choice", "required": True, "default": None, "description": "",
         "choices": ["formal", "friendly"], "min_length": None, "max_length": None, "minimum": None, "maximum": None},
    ]


def test_prompt_template_store_creates_local_versioned_template_with_private_permissions(tmp_path: Path):
    store = PromptTemplateStore(tmp_path)
    record = store.create(
        name="Research brief",
        description="Generate a sourced short brief.",
        body="Write a {{tone}} brief for {{name}} with {{count}} sections. Sources={{include_sources}} temp={{temperature}}",
        variables=_variables(),
    )
    assert record["version"] == 1
    assert record["name"] == "Research brief"
    assert record["content_sha256"]
    assert store.versions(record["id"]) == [{
        "version": 1, "updated_at": record["updated_at"], "name": "Research brief",
        "content_sha256": record["content_sha256"], "current": True,
    }]
    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_prompt_template_render_validates_typed_values_and_never_executes_expressions(tmp_path: Path):
    store = PromptTemplateStore(tmp_path)
    record = store.create(
        name="Research brief", body="Hello {{name}}; count={{count}}; temp={{temperature}}; sources={{include_sources}}; tone={{tone}}",
        variables=_variables(),
    )
    result = store.render(record["id"], {"name": "Alex", "tone": "friendly"})
    assert result is not None
    assert result["rendered"] == "Hello Alex; count=3; temp=0.2; sources=true; tone=friendly"
    assert result["execution_performed"] is False
    assert result["model_invoked"] is False
    assert "__import__" not in result["rendered"]


def test_prompt_template_render_rejects_missing_wrong_typed_out_of_range_and_extra_values():
    variables = _variables()
    with pytest.raises(PromptTemplateError, match="missing required"):
        render_prompt("{{name}}/{{tone}}", variables, {})
    with pytest.raises(PromptTemplateError, match="must be a string"):
        render_prompt("{{name}}", variables, {"name": 42, "tone": "formal"})
    with pytest.raises(PromptTemplateError, match="integer"):
        render_prompt("{{count}}", variables, {"name": "Alex", "tone": "formal", "count": True})
    with pytest.raises(PromptTemplateError, match="minimum"):
        render_prompt("{{count}}", variables, {"name": "Alex", "tone": "formal", "count": 0})
    with pytest.raises(PromptTemplateError, match="unknown variables"):
        render_prompt("{{name}}", variables, {"name": "Alex", "tone": "formal", "shell": "ignored"})


def test_prompt_template_rejects_unknown_or_malformed_placeholders_and_duplicate_variables(tmp_path: Path):
    store = PromptTemplateStore(tmp_path)
    with pytest.raises(PromptTemplateError, match="undefined variable"):
        store.create(name="Bad", body="Hello {{unknown}}", variables=[])
    with pytest.raises(PromptTemplateError, match="only {{variable_name}} syntax"):
        store.create(name="Bad", body="Hello {{name | exec}}", variables=_variables())
    duplicate = _variables()[:2] + [_variables()[0]]
    with pytest.raises(PromptTemplateError, match="unique"):
        store.create(name="Bad", body="{{name}}", variables=duplicate)


def test_prompt_template_update_uses_optimistic_version_and_retains_revision_history(tmp_path: Path):
    store = PromptTemplateStore(tmp_path)
    record = store.create(name="Template", body="Hello {{name}}", variables=[_variables()[0]])
    updated = store.update(record["id"], expected_version=1, name="Template v2", body="Hi {{name}}", variables=[_variables()[0]])
    assert updated is not None
    assert updated["version"] == 2
    history = store.versions(record["id"])
    assert [item["version"] for item in history] == [2, 1]
    previous = store.get_revision(record["id"], 1)
    assert previous["body"] == "Hello {{name}}"
    rendered_previous = store.render(record["id"], {"name": "Taylor"}, version=1)
    assert rendered_previous["rendered"] == "Hello Taylor"
    with pytest.raises(PromptTemplateConflict, match="changed since it was loaded"):
        store.update(record["id"], expected_version=1, name="Stale", body="Stale")
    with pytest.raises(PromptTemplateConflict, match="changed since it was loaded"):
        store.delete(record["id"], expected_version=1)


def test_prompt_template_store_rejects_duplicate_name_tampering_and_invalid_version(tmp_path: Path):
    store = PromptTemplateStore(tmp_path)
    one = store.create(name="Same", body="A prompt")
    with pytest.raises(PromptTemplateError, match="already exists"):
        store.create(name="same", body="A different prompt")
    document = json.loads(store.path.read_text(encoding="utf-8"))
    document["templates"][0]["body"] = "tampered"
    store.path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PromptTemplateError, match="fingerprint"):
        store.list_page()
    with pytest.raises(PromptTemplateError, match="revision"):
        store.get_revision(one["id"], 0)


def test_prompt_template_delete_requires_current_version_and_symlinks_are_rejected(tmp_path: Path):
    store = PromptTemplateStore(tmp_path)
    record = store.create(name="Delete me", body="Prompt")
    assert store.delete(record["id"], expected_version=1)
    assert store.get(record["id"]) is None

    workspace = tmp_path / "linked-workspace"
    workspace.mkdir()
    metadata = workspace / ".axiom"
    metadata.mkdir()
    source = tmp_path / "real.json"
    source.write_text('{"schema_version":1,"templates":[]}', encoding="utf-8")
    (metadata / "prompt-templates.json").symlink_to(source)
    with pytest.raises(PromptTemplateError, match="symbolic links"):
        PromptTemplateStore(workspace).list_page()
