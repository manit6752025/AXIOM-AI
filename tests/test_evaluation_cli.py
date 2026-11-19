import json

from typer.testing import CliRunner

from axiom.cli.main import app
from axiom.evaluation.store import EvaluationStore


def test_cli_runs_lists_shows_and_deletes_offline_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "evaluation.jsonl"
    data.write_text(
        '{"prediction":"answer","reference":"answer"}\n'
        '{"prediction":"wrong","reference":"right"}\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, [
        "evaluation", "run", str(data),
        "--prediction-field", "prediction",
        "--reference-field", "reference",
        "--json",
    ])
    assert result.exit_code == 0, result.stdout
    output = json.loads(result.stdout)
    saved = output["run"]
    assert saved["execution_status"] == "precomputed_outputs_only"
    assert saved["report"]["metrics"]["exact_match"] == 0.5
    assert EvaluationStore().get(saved["id"]) == saved

    listed = runner.invoke(app, ["evaluation", "list", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)["total"] == 1

    shown = runner.invoke(app, ["evaluation", "show", saved["id"]])
    assert shown.exit_code == 0, shown.stdout
    assert json.loads(shown.stdout)["id"] == saved["id"]

    deleted = runner.invoke(app, ["evaluation", "delete", saved["id"], "--yes"])
    assert deleted.exit_code == 0, deleted.stdout
    assert json.loads(deleted.stdout)["deleted"]["id"] == saved["id"]
