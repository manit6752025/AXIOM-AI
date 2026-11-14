import json

from typer.testing import CliRunner

from axiom.cli.main import app
from axiom.core.workflows import WorkflowStore


def test_cli_lists_shows_and_validates_workflows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = WorkflowStore()
    workflow = {
        "id": "workflow-1",
        "name": "Dataset plan",
        "created_at": "2026-10-19T00:00:00Z",
        "updated_at": "2026-10-19T00:00:00Z",
        "nodes": [
            {"id": "dataset", "type": "dataset.profile"},
            {"id": "plan", "type": "training.plan"},
        ],
        "edges": [{"source": "dataset", "target": "plan"}],
    }
    store.save(workflow)
    runner = CliRunner()

    listed = runner.invoke(app, ["workflow", "list", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)[0]["id"] == "workflow-1"

    shown = runner.invoke(app, ["workflow", "show", "workflow-1", "--json"])
    assert shown.exit_code == 0, shown.stdout
    assert json.loads(shown.stdout)["execution_status"] == "not_executable"

    validated = runner.invoke(app, ["workflow", "validate", "workflow-1"])
    assert validated.exit_code == 0, validated.stdout
    assert json.loads(validated.stdout)["valid"] is True


def test_cli_validation_rejects_a_cyclic_saved_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    WorkflowStore().save({
        "id": "cycle",
        "name": "Cycle",
        "created_at": "now",
        "updated_at": "now",
        "nodes": [
            {"id": "first", "type": "dataset.profile"},
            {"id": "second", "type": "model.inspect"},
        ],
        "edges": [
            {"source": "first", "target": "second"},
            {"source": "second", "target": "first"},
        ],
    })

    result = CliRunner().invoke(app, ["workflow", "validate", "cycle"])
    assert result.exit_code == 1
    assert "acyclic" in result.stdout


def test_cli_workflow_page_json_and_cursor_search_are_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = WorkflowStore()
    for workflow_id, name in (
        ("first", "First audit"),
        ("second", "Second training"),
        ("third", "Third audit"),
    ):
        store.save({
            "id": workflow_id,
            "name": name,
            "created_at": "2026-11-14T00:00:00Z",
            "updated_at": "2026-11-14T00:00:00Z",
            "nodes": [{"id": "step", "type": "dataset.profile"}],
            "edges": [],
        })

    runner = CliRunner()
    first = runner.invoke(app, ["workflow", "list", "--limit", "2", "--page-json"])
    assert first.exit_code == 0, first.output
    first_page = json.loads(first.stdout)
    assert [item["id"] for item in first_page["workflows"]] == ["third", "second"]
    assert first_page["next_cursor"] == "second"
    assert first_page["has_more"] is True

    second = runner.invoke(app, [
        "workflow", "list", "--limit", "2", "--after", first_page["next_cursor"], "--json",
    ])
    assert second.exit_code == 0, second.output
    assert [item["id"] for item in json.loads(second.stdout)] == ["first"]

    filtered = runner.invoke(app, ["workflow", "list", "--search", "audit", "--page-json"])
    assert filtered.exit_code == 0, filtered.output
    assert filtered.stdout
    assert json.loads(filtered.stdout)["total"] == 2
