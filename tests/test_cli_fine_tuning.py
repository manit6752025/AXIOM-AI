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
    assert "cancel" in help_result.output


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

def test_train_cancel_requests_cooperative_stop_and_can_print_json(monkeypatch):
    import json
    from axiom.cli import fine_tuning as cli_fine_tuning

    expected = {
        "id": "a" * 32,
        "status": "cancelling",
        "cancellation": {
            "supported": True,
            "requested": True,
            "requestable": False,
            "mode": "cooperative",
            "boundary": "between training steps",
            "unavailable_reason": "already_requested",
        },
    }
    observed = []
    monkeypatch.setattr(cli_fine_tuning, "cancel_finetuning_job", lambda job_id: observed.append(job_id) or expected)
    result = CliRunner().invoke(app, ["--no-banner", "train", "cancel", "a" * 32, "--json"])
    assert result.exit_code == 0, result.output
    assert observed == ["a" * 32]
    assert json.loads(result.stdout) == expected


def test_train_cancel_reports_when_job_is_already_finalizing(monkeypatch):
    from axiom.cli import fine_tuning as cli_fine_tuning

    expected = {
        "id": "b" * 32,
        "status": "finalizing",
        "cancellation": {"unavailable_reason": "finalization_started"},
    }
    monkeypatch.setattr(cli_fine_tuning, "cancel_finetuning_job", lambda _job_id: expected)
    result = CliRunner().invoke(app, ["--no-banner", "train", "cancel", "b" * 32])
    assert result.exit_code == 0, result.output
    assert "finalization_started" in result.output
