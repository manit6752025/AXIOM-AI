from __future__ import annotations

from fastapi.testclient import TestClient

import axiom.api.prediction_jobs as prediction_api
from axiom.api.server import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prediction_api, "active_workspace_root", lambda: tmp_path)
    return TestClient(create_app())


def _summary(status="running"):
    return {
        "id": "a" * 32,
        "status": status,
        "created_at": "2026-11-21T00:00:00+00:00",
        "started_at": "2026-11-21T00:00:01+00:00",
        "finished_at": None,
        "dataset_path": "data/eval.jsonl",
        "output_path": "data/predictions.jsonl",
        "artifact_ready": False,
        "rows_total": 8,
        "rows_completed": 4,
        "prompts_truncated": 1,
        "evaluation_run_ids": [],
        "error_code": None,
        "scope": {
            "prompts_returned": False,
            "predictions_returned": False,
            "logs_returned": False,
            "absolute_paths_returned": False,
            "output_file_published_only_after_completion": True,
            "job_state_persisted": True,
            "server_restart_clears_job_state": False,
            "generation_resumed_after_restart": False,
        },
    }


def test_prediction_job_api_starts_lists_reads_and_cancels_workspace_jobs(tmp_path, monkeypatch):
    started = {}
    summary = _summary()

    def fake_start(root, **kwargs):
        started.update(root=root, **kwargs)
        return summary

    monkeypatch.setattr(prediction_api, "start_prediction_job", fake_start)
    monkeypatch.setattr(prediction_api, "list_prediction_jobs", lambda root, limit: [summary])
    monkeypatch.setattr(prediction_api, "get_prediction_job", lambda job_id, root: summary if job_id == "a" * 32 else None)
    monkeypatch.setattr(prediction_api, "cancel_prediction_job", lambda job_id, root: {
        "job": {**summary, "status": "cancelling"},
        "cancellation_accepted": True,
        "note": "Cancellation is cooperative.",
    })

    with _client(tmp_path, monkeypatch) as client:
        started_response = client.post("/api/prediction-jobs", json={
            "model_path": "models/qwen",
            "dataset_path": "data/eval.jsonl",
            "output_path": "data/predictions.jsonl",
            "max_new_tokens": 32,
            "device": "cpu",
            "precision": "fp32",
        })
        assert started_response.status_code == 202, started_response.text
        assert started_response.json()["job"]["id"] == "a" * 32
        assert started_response.json()["limits"]["rows"] == 100
        assert started["root"] == tmp_path
        assert started["dataset_path"] == "data/eval.jsonl"
        assert started["device"] == "cpu"

        page = client.get("/api/prediction-jobs", params={"limit": 5})
        assert page.status_code == 200
        assert page.json()["jobs"][0]["rows_completed"] == 4

        inspected = client.get(f"/api/prediction-jobs/{'a' * 32}")
        assert inspected.status_code == 200
        assert inspected.json()["job"]["prompts_truncated"] == 1

        cancelled = client.post(f"/api/prediction-jobs/{'a' * 32}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["cancellation_accepted"] is True

        missing = client.get(f"/api/prediction-jobs/{'b' * 32}")
        assert missing.status_code == 404


def test_prediction_job_api_rejects_unbounded_and_unknown_fields(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        invalid = client.post("/api/prediction-jobs", json={
            "model_path": "models/qwen",
            "dataset_path": "data/eval.jsonl",
            "output_path": "data/out.jsonl",
            "max_new_tokens": 65,
        })
        extra = client.post("/api/prediction-jobs", json={
            "model_path": "models/qwen",
            "dataset_path": "data/eval.jsonl",
            "output_path": "data/out.jsonl",
            "shell": "touch /tmp/owned",
        })
        assert invalid.status_code == 422
        assert extra.status_code == 422
