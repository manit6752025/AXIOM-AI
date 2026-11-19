from __future__ import annotations

from fastapi.testclient import TestClient

from axiom.api.server import create_app
from axiom.core import model_studio as studio


def test_model_studio_cancel_endpoint_returns_cancelling_job(monkeypatch):
    expected = {
        "schema_version": 1,
        "id": "a" * 32,
        "status": "cancelling",
        "cancel_requested": True,
        "operation": "pull_native",
    }
    observed = {}

    def cancel(job_id: str):
        observed["job_id"] = job_id
        return expected

    monkeypatch.setattr(studio, "cancel_job", cancel)
    response = TestClient(create_app()).post(f"/api/models/studio/jobs/{'a' * 32}/cancel")
    assert response.status_code == 200
    assert response.json() == expected
    assert observed == {"job_id": "a" * 32}


def test_model_studio_cancel_endpoint_distinguishes_missing_invalid_and_unavailable(monkeypatch):
    client = TestClient(create_app())

    monkeypatch.setattr(studio, "cancel_job", lambda _job_id: None)
    missing = client.post(f"/api/models/studio/jobs/{'b' * 32}/cancel")
    assert missing.status_code == 404

    def invalid(_job_id: str):
        raise ValueError("Invalid Model Studio job ID.")

    monkeypatch.setattr(studio, "cancel_job", invalid)
    invalid_response = client.post("/api/models/studio/jobs/not-valid/cancel")
    assert invalid_response.status_code == 422

    def unavailable(_job_id: str):
        raise studio.ModelStudioError("The Model Studio worker is not available to accept cancellation.")

    monkeypatch.setattr(studio, "cancel_job", unavailable)
    conflict = client.post(f"/api/models/studio/jobs/{'c' * 32}/cancel")
    assert conflict.status_code == 409
