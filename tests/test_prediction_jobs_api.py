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

def test_prediction_job_api_resumes_only_when_core_verification_succeeds(tmp_path, monkeypatch):
    resumed = {}
    summary = {**_summary("queued"), "resumable": False, "resumption_count": 1}

    def fake_resume(job_id, root):
        resumed.update(job_id=job_id, root=root)
        return summary

    monkeypatch.setattr(prediction_api, "resume_prediction_job", fake_resume)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(f"/api/prediction-jobs/{'a' * 32}/resume")
        assert response.status_code == 202, response.text
        assert response.json()["job"]["status"] == "queued"
        assert response.json()["scope"] == {
            "dataset_fingerprint_verified": True,
            "model_and_adapter_fingerprints_verified": True,
            "checkpoint_identity_and_digest_verified": True,
            "partial_output_published": False,
            "prompts_returned": False,
            "predictions_returned": False,
        }
        assert resumed == {"job_id": "a" * 32, "root": tmp_path}


def test_prediction_job_api_reports_resume_validation_failures(tmp_path, monkeypatch):
    def invalid_resume(job_id, root):
        raise ValueError("The original prediction dataset fingerprint changed.")

    monkeypatch.setattr(prediction_api, "resume_prediction_job", invalid_resume)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(f"/api/prediction-jobs/{'a' * 32}/resume")
        assert response.status_code == 422
        assert "fingerprint changed" in response.json()["detail"]

        monkeypatch.setattr(prediction_api, "resume_prediction_job", lambda job_id, root: None)
        missing = client.post(f"/api/prediction-jobs/{'b' * 32}/resume")
        assert missing.status_code == 404

