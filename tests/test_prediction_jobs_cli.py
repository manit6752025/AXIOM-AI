from __future__ import annotations

import json

from typer.testing import CliRunner

import axiom.cli.engine as engine_cli


runner = CliRunner()


def test_engine_prediction_job_cli_starts_and_lists_persisted_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_cli, "active_workspace_root", lambda: tmp_path)
    summary = {
        "id": "c" * 32,
        "status": "completed",
        "dataset_path": "data/eval.jsonl",
        "output_path": "data/predictions.jsonl",
        "artifact_ready": True,
        "rows_total": 5,
        "rows_completed": 5,
        "prompts_truncated": 1,
        "error_code": None,
    }
    captured = {}
    monkeypatch.setattr(engine_cli, "start_prediction_job", lambda root, **kwargs: captured.update(root=root, **kwargs) or {**summary, "status": "queued", "rows_completed": 0, "artifact_ready": False})
    monkeypatch.setattr(engine_cli, "wait_prediction_job", lambda job_id, root, timeout_seconds: {
        "job": summary, "wait_outcome": "completed", "timed_out": False, "waited_seconds": 0.1,
    })
    monkeypatch.setattr(engine_cli, "list_prediction_jobs", lambda root, limit: [summary])
    monkeypatch.setattr(engine_cli, "get_prediction_job", lambda job_id, root: summary if job_id == summary["id"] else None)

    started = runner.invoke(engine_cli.engine_app, [
        "prediction", "start",
        "--model-path", "models/qwen",
        "--dataset-path", "data/eval.jsonl",
        "--output-path", "data/predictions.jsonl",
        "--device", "cpu",
        "--json",
    ])
    assert started.exit_code == 0, started.output
    assert json.loads(started.stdout)["job"]["id"] == summary["id"]
    assert captured["root"] == tmp_path
    assert captured["device"] == "cpu"

    listed = runner.invoke(engine_cli.engine_app, ["prediction", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)["jobs"][0]["rows_completed"] == 5

    shown = runner.invoke(engine_cli.engine_app, ["prediction", "show", summary["id"], "--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["job"]["id"] == summary["id"]


def test_engine_prediction_job_cli_cancels_cooperatively(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_cli, "active_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(engine_cli, "cancel_prediction_job", lambda job_id, root: {
        "job": {"id": job_id, "status": "cancelling"},
        "cancellation_accepted": True,
        "note": "Cancellation is cooperative.",
    })
    result = runner.invoke(engine_cli.engine_app, ["prediction", "cancel", "d" * 32, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["cancellation_accepted"] is True

def test_engine_prediction_job_cli_resumes_from_verified_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_cli, "active_workspace_root", lambda: tmp_path)
    summary = {
        "id": "e" * 32,
        "status": "queued",
        "rows_total": 8,
        "rows_completed": 4,
        "resumable": False,
        "resumption_count": 1,
    }
    captured = {}
    monkeypatch.setattr(
        engine_cli,
        "resume_prediction_job",
        lambda job_id, root: captured.update(job_id=job_id, root=root) or summary,
    )

    result = runner.invoke(engine_cli.engine_app, ["prediction", "resume", summary["id"], "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["job"]["resumption_count"] == 1
    assert captured == {"job_id": summary["id"], "root": tmp_path}


def test_engine_prediction_job_cli_rejects_unavailable_resume_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_cli, "active_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(
        engine_cli,
        "resume_prediction_job",
        lambda job_id, root: (_ for _ in ()).throw(ValueError("No verified checkpoint is available.")),
    )
    result = runner.invoke(engine_cli.engine_app, ["prediction", "resume", "e" * 32])
    assert result.exit_code != 0
    assert "could not be resumed safely" in result.output

