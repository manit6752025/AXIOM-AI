from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.testing import CliRunner

from axiom.cli.prompt_templates import register


def _app() -> typer.Typer:
    app = typer.Typer()
    register(app)
    return app


def test_prompt_template_cli_create_list_show_render_and_revision_history(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    app = _app()

    created = runner.invoke(app, [
        "prompt", "create", "Research brief",
        "--body", "Write a {{tone}} brief for {{name}}.",
        "--description", "A short sourced brief.",
        "--variables-json", '[{"name":"tone","type":"choice","choices":["formal","friendly"]},{"name":"name","type":"string"}]',
        "--json",
    ])
    assert created.exit_code == 0, created.stdout
    record = json.loads(created.stdout)
    template_id = record["id"]
    assert record["version"] == 1

    listed = runner.invoke(app, ["prompt", "list", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert listed.stdout
    assert json.loads(listed.stdout)["templates"][0]["id"] == template_id

    shown = runner.invoke(app, ["prompt", "show", template_id, "--json"])
    assert shown.exit_code == 0, shown.stdout
    assert json.loads(shown.stdout)["body"] == "Write a {{tone}} brief for {{name}}."

    rendered = runner.invoke(app, [
        "prompt", "render", template_id,
        "--values-json", '{"tone":"friendly","name":"Taylor"}', "--json",
    ])
    assert rendered.exit_code == 0, rendered.stdout
    result = json.loads(rendered.stdout)
    assert result["rendered"] == "Write a friendly brief for Taylor."
    assert result["model_invoked"] is False
    assert result["execution_performed"] is False

    updated = runner.invoke(app, [
        "prompt", "update", template_id, "--expected-version", "1",
        "--name", "Research brief v2", "--body", "Prepare a {{tone}} brief for {{name}}.",
        "--variables-json", '[{"name":"tone","type":"choice","choices":["formal","friendly"]},{"name":"name","type":"string"}]',
        "--json",
    ])
    assert updated.exit_code == 0, updated.stdout
    assert json.loads(updated.stdout)["version"] == 2

    versions = runner.invoke(app, ["prompt", "versions", template_id, "--json"])
    assert versions.exit_code == 0, versions.stdout
    assert [item["version"] for item in json.loads(versions.stdout)["versions"]] == [2, 1]

    denied = runner.invoke(app, ["prompt", "delete", template_id, "--expected-version", "2"])
    assert denied.exit_code == 2

    deleted = runner.invoke(app, ["prompt", "delete", template_id, "--expected-version", "2", "--yes"])
    assert deleted.exit_code == 0, deleted.stdout
    assert "Deleted" in deleted.stdout


def test_prompt_template_cli_enforces_core_text_limits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    app = _app()

    too_long_name = runner.invoke(app, [
        "prompt", "create", "N" * 81, "--body", "Prompt", "--json",
    ])
    assert too_long_name.exit_code != 0
    assert not (tmp_path / ".axiom" / "prompt-templates.json").exists()

    too_long_description = runner.invoke(app, [
        "prompt", "create", "Valid name", "--body", "Prompt",
        "--description", "D" * 241, "--json",
    ])
    assert too_long_description.exit_code != 0
    assert not (tmp_path / ".axiom" / "prompt-templates.json").exists()


def test_prompt_template_cli_reads_bounded_regular_body_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    body_file = tmp_path / "prompt.txt"
    body_file.write_text("Summarize {{topic}}", encoding="utf-8")
    runner = CliRunner()
    app = _app()

    created = runner.invoke(app, [
        "prompt", "create", "File prompt", "--body-file", str(body_file),
        "--variables-json", '[{"name":"topic","type":"string"}]', "--json",
    ])
    assert created.exit_code == 0, created.stdout

    linked = tmp_path / "linked.txt"
    linked.symlink_to(body_file)
    rejected = runner.invoke(app, [
        "prompt", "create", "Linked prompt", "--body-file", str(linked), "--json",
    ])
    assert rejected.exit_code != 0
