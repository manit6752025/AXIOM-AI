from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from axiom.api.server import create_app
from axiom.core.request_admission import (
    RequestAdmissionCancelled,
    RequestAdmissionQueueFull,
    RequestAdmissionTimeout,
)


@pytest.mark.parametrize(
    ("kind", "exception", "status", "code"),
    [
        ("full", RequestAdmissionQueueFull, 429, "request_queue_full"),
        ("timeout", RequestAdmissionTimeout, 504, "request_queue_timeout"),
        ("cancelled", RequestAdmissionCancelled, 499, "request_cancelled"),
    ],
)
def test_application_maps_shared_native_admission_failures_to_http(
    kind: str, exception: type[Exception], status: int, code: str,
):
    app = create_app()

    @app.get(f"/api/test/admission/{kind}")
    def fail_admission():
        raise exception("Synthetic admission failure for error-contract testing.")

    with TestClient(app) as client:
        response = client.get(f"/api/test/admission/{kind}")

    assert response.status_code == status
    assert response.headers.get("x-request-id")
    assert response.json()["error"]["code"] == code
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "Synthetic admission failure" in response.json()["detail"]
    if status == 429:
        assert response.headers["retry-after"] == "1"
    else:
        assert "retry-after" not in response.headers
