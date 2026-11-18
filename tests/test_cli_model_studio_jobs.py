from __future__ import annotations

import json

from typer.testing import CliRunner

from axiom.cli.model_studio import studio_app


def test_model_studio_jobs_cli_supports_offset_search_and_json(monkeypatch):
    from axiom.cli import model_studio

    expected = {
        "jobs": [{"id": "a" * 32, "operation": "pull_native", "status": "completed", "updated_at": "2026-11-18"}],
        "total": 9, "limit": 3, "offset": 6, "has_more": False,
        "truncated": True, "scan_limit": 5000, "filters": {"search": "qwen"},
    }
    observed = {}
    def fake_page(*, limit, offset, search):
        observed.update(limit=limit, offset=offset, search=search)
        return expected
    monkeypatch.setattr(model_studio.studio, "list_jobs_page", fake_page)
    result = CliRunner().invoke(
        studio_app,
        ["jobs", "--limit", "3", "--offset", "6", "--search", "qwen", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert observed == {"limit": 3, "offset": 6, "search": "qwen"}
    assert json.loads(result.stdout) == expected


def test_model_studio_native_download_receives_progress_and_cancel_callbacks(monkeypatch):
    import threading

    from axiom.cli import model_studio

    observed = {}

    def fake_pull(repo_id, *, on_log=None, on_progress=None, cancel_event=None):
        observed.update(
            repo_id=repo_id,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )
        assert isinstance(cancel_event, threading.Event)
        assert not cancel_event.is_set()
        return {"operation": "pull_native", "path": "/models/qwen"}

    monkeypatch.setattr(model_studio.studio, "pull_native_model", fake_pull)
    result = CliRunner().invoke(studio_app, ["pull-native", "Qwen/test", "--json"])

    assert result.exit_code == 0, result.output
    assert observed["repo_id"] == "Qwen/test"
    assert isinstance(observed["cancel_event"], threading.Event)
    assert json.loads(result.stdout)["path"] == "/models/qwen"


def test_model_studio_native_download_reports_cooperative_cancel(monkeypatch):
    from axiom.cli import model_studio

    def cancelled(repo_id, *, on_log=None, on_progress=None, cancel_event=None):
        cancel_event.set()
        raise model_studio.studio.ModelStudioCancelled("simulated cancellation")

    monkeypatch.setattr(model_studio.studio, "pull_native_model", cancelled)
    result = CliRunner().invoke(studio_app, ["pull-native", "Qwen/test"])

    assert result.exit_code == 130
    assert "cancelled" in result.output.lower()
