from __future__ import annotations

from typer.testing import CliRunner
from typer.main import get_command

from axiom.cli.main import app


def test_train_help_exposes_real_fine_tuning_commands():
    runner = CliRunner()
    help_result = runner.invoke(app, ["--no-banner", "train", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "run" in help_result.output
    assert "jobs" in help_result.output
    assert "status" in help_result.output


def test_train_run_exposes_required_input_options():
    command = get_command(app).commands["train"].commands["run"]
    options = {
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", ())
    }
    assert {"--model-path", "--dataset", "--output-dir"} <= options


def test_train_jobs_supports_offset_and_page_json(monkeypatch):
    import json
    from axiom.cli import fine_tuning as cli_fine_tuning

    expected = {
        "jobs": [{"id": "a" * 32, "status": "completed"}],
        "total": 8, "limit": 2, "offset": 4, "has_more": True,
        "truncated": False, "scan_limit": 5000,
    }
    observed = {}
    def fake_page(*, limit, offset):
        observed.update(limit=limit, offset=offset)
        return expected
    monkeypatch.setattr(cli_fine_tuning, "list_jobs_page", fake_page)
    result = CliRunner().invoke(app, ["--no-banner", "train", "jobs", "--limit", "2", "--offset", "4", "--json"])
    assert result.exit_code == 0, result.output
    assert observed == {"limit": 2, "offset": 4}
    assert json.loads(result.stdout) == expected
