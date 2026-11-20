from __future__ import annotations

from fastapi.testclient import TestClient

from axiom.api.server import create_app


def test_fine_tuning_runtime_route_is_registered(monkeypatch):
    from axiom.api import fine_tuning
    monkeypatch.setattr(fine_tuning, "runtime_status", lambda: {
        "backend": "transformers-peft-local-lora", "can_start": False,
        "packages": {"torch": False, "transformers": False, "peft": False, "accelerate": False},
        "model_loading": "local_files_only", "scope_note": "Local only.",
    })
    response = TestClient(create_app()).get("/api/fine-tuning/runtime")
    assert response.status_code == 200
    assert response.json()["model_loading"] == "local_files_only"
    assert response.json()["can_start"] is False


def test_fine_tuning_start_route_returns_created_job(monkeypatch):
    from axiom.api import fine_tuning
    expected = {"id": "a" * 32, "status": "queued", "logs": ["Job queued."]}
    monkeypatch.setattr(fine_tuning, "start_finetuning_job", lambda _config: expected)
    payload = {
        "model_path": "/models/base", "dataset_path": "/data/train.jsonl", "output_dir": "/runs/adapter",
        "text_field": "text", "max_steps": 3, "sequence_length": 64,
        "learning_rate": 0.0001, "lora_rank": 4,
    }
    response = TestClient(create_app()).post("/api/fine-tuning/jobs", json=payload)
    assert response.status_code == 202
    assert response.json()["id"] == expected["id"]
    assert response.json()["status"] == "queued"


def test_fine_tuning_start_rejects_out_of_bounds_job(monkeypatch):
    from axiom.api import fine_tuning
    payload = {
        "model_path": "/models/base", "dataset_path": "/data/train.jsonl", "output_dir": "/runs/adapter",
        "text_field": "text", "max_steps": 1001, "sequence_length": 64,
        "learning_rate": 0.0001, "lora_rank": 4,
    }
    response = TestClient(create_app()).post("/api/fine-tuning/jobs", json=payload)
    assert response.status_code == 422


def test_fine_tuning_start_rejects_when_a_job_is_already_active(monkeypatch):
    from axiom.api import fine_tuning
    from axiom.core.fine_tuning import ActiveFineTuningJobError
    monkeypatch.setattr(fine_tuning, "start_finetuning_job", lambda _config: (_ for _ in ()).throw(ActiveFineTuningJobError("active")))
    payload = {
        "model_path": "/models/base", "dataset_path": "/data/train.jsonl", "output_dir": "/runs/adapter",
        "text_field": "text", "max_steps": 3, "sequence_length": 64,
        "learning_rate": 0.0001, "lora_rank": 4,
    }
    response = TestClient(create_app()).post("/api/fine-tuning/jobs", json=payload)
    assert response.status_code == 409


def test_fine_tuning_jobs_endpoint_returns_bounded_page(monkeypatch):
    from axiom.api import fine_tuning

    expected = {
        "jobs": [{"id": "a" * 32, "status": "completed"}],
        "total": 11, "limit": 5, "offset": 5, "has_more": True,
        "truncated": False, "scan_limit": 5000,
    }
    monkeypatch.setattr(fine_tuning, "list_jobs_page", lambda **kwargs: expected)
    response = TestClient(create_app()).get("/api/fine-tuning/jobs?limit=5&offset=5")
    assert response.status_code == 200
    assert response.json() == expected


def test_fine_tuning_jobs_endpoint_rejects_unbounded_paging():
    response = TestClient(create_app()).get("/api/fine-tuning/jobs?limit=101")
    assert response.status_code == 422

def test_fine_tuning_cancel_route_returns_explicit_cancellation_contract(monkeypatch):
    from axiom.api import fine_tuning

    expected = {
        "id": "a" * 32,
        "status": "cancelling",
        "cancellation": {
            "supported": True,
            "requested": True,
            "requestable": False,
            "mode": "cooperative",
            "boundary": "before worker start or between training steps; adapter finalization is not interruptible",
            "unavailable_reason": "already_requested",
        },
    }
    monkeypatch.setattr(fine_tuning, "cancel_finetuning_job", lambda _job_id: expected)
    response = TestClient(create_app()).post(f"/api/fine-tuning/jobs/{'a' * 32}/cancel")
    assert response.status_code == 200
    assert response.json() == expected


def test_fine_tuning_cancel_route_returns_not_found_for_missing_job(monkeypatch):
    from axiom.api import fine_tuning

    monkeypatch.setattr(fine_tuning, "cancel_finetuning_job", lambda _job_id: None)
    response = TestClient(create_app()).post(f"/api/fine-tuning/jobs/{'b' * 32}/cancel")
    assert response.status_code == 404
