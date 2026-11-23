from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk" / "python"))

from axiom_sdk import (  # noqa: E402
    AxiomAPIError,
    AxiomClient,
    AxiomConfigurationError,
    AxiomNetworkError,
)


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    redirect_target: str | None = None

    def log_message(self, _format, *_args):
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        parsed = urlsplit(self.path)
        body = self._read_body()
        type(self).requests.append({
            "method": "GET",
            "path": parsed.path,
            "query": parse_qs(parsed.query),
            "headers": dict(self.headers.items()),
            "body": body,
        })
        if parsed.path == "/api/health":
            self._send(200, {"status": "ok", "version": "test"})
        elif parsed.path == "/api/meta":
            self._send(200, {
                "api_contract_version": 1,
                "application_version": "test",
                "openapi_url": "/api/openapi.json",
                "compatibility_policy": "Additive v1.",
            })
        elif parsed.path == "/api/workspace/models":
            self._send(200, {"models": [], "total": 0})
        elif parsed.path == "/api/prediction-jobs":
            self._send(200, {"schema_version": 1, "jobs": [{"id": "a" * 32, "status": "interrupted", "resumable": True}], "limit": 20})
        elif parsed.path == f"/api/prediction-jobs/{'a' * 32}":
            self._send(200, {"schema_version": 1, "job": {"id": "a" * 32, "status": "interrupted", "resumable": True}})
        elif parsed.path == "/api/workspaces/ws-1/backup":
            self._send_raw(200, b"PK\x03\x04backup", "application/zip")
        elif parsed.path == "/api/redirect":
            self.send_response(302)
            self.send_header("Location", type(self).redirect_target or "/api/health")
            self.end_headers()
        elif parsed.path == "/api/error":
            self._send(404, {
                "request_id": "body-request-id",
                "error": {
                    "code": "workspace_not_found",
                    "message": "Workspace not found.",
                    "details": {"workspace_id": "missing"},
                },
            })
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        parsed = urlsplit(self.path)
        body = self._read_body()
        type(self).requests.append({
            "method": "POST",
            "path": parsed.path,
            "query": parse_qs(parsed.query),
            "headers": dict(self.headers.items()),
            "body": body,
        })
        if parsed.path == "/api/prediction-jobs":
            self._send(202, {"schema_version": 1, "job": {"id": "a" * 32, "status": "queued", "resumable": False}, "limits": {}, "scope": {}})
        elif parsed.path == f"/api/prediction-jobs/{'a' * 32}/resume":
            self._send(202, {"schema_version": 1, "job": {"id": "a" * 32, "status": "queued", "resumption_count": 1}, "scope": {"checkpoint_identity_and_digest_verified": True, "generation_resumed_after_restart": True}})
        elif parsed.path == f"/api/prediction-jobs/{'a' * 32}/cancel":
            self._send(200, {"schema_version": 1, "job": {"id": "a" * 32, "status": "cancelling"}, "cancellation_accepted": True, "note": "Cancellation requested."})
        if parsed.path == "/api/workspaces/backup/restore":
            self._send(201, {
                "schema_version": 1,
                "workspace": {"id": "ws-restored", "name": parse_qs(parsed.query).get("name", [""])[0]},
                "backup": {"integrity": "verified"},
            })
        elif parsed.path == "/api/engine/telemetry-history":
            self._send(409, {"detail": "Persistent telemetry is disabled."})
        elif parsed.path == "/api/workspaces":
            self._send(201, {"workspace": {"id": "ws-new", "name": "Test"}})
        else:
            self._send(404, {"detail": "not found"})

    def _send(self, status: int, payload: dict):
        self._send_raw(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_raw(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", "response-request-id")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def api_server():
    _Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Handler.redirect_target = None
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_python_sdk_sends_request_id_auth_and_workspace_headers(api_server):
    with AxiomClient(
        api_server,
        token="read-token",
        workspace_id="ws-123",
        headers={"X-Client-Name": "sdk-test"},
    ) as client:
        assert client.health()["status"] == "ok"
        assert client.meta()["api_contract_version"] == 1
        client.models(limit=12, offset=3, search="llama")

    health, meta, models = _Handler.requests
    for record in (health, meta, models):
        headers = {name.casefold(): value for name, value in record["headers"].items()}
        assert len(headers["x-request-id"]) == 32
        assert headers["authorization"] == "Bearer read-token"
        assert headers["x-axiom-workspace-id"] == "ws-123"
        assert headers["x-client-name"] == "sdk-test"
    assert models["query"] == {"limit": ["12"], "offset": ["3"], "search": ["llama"]}


def test_python_sdk_parses_structured_api_errors(api_server):
    with AxiomClient(api_server) as client:
        with pytest.raises(AxiomAPIError) as raised:
            client.request_json("GET", "/api/error")
    error = raised.value
    assert error.status_code == 404
    assert error.code == "workspace_not_found"
    assert error.request_id == "response-request-id"
    assert error.details == {"workspace_id": "missing"}


def test_python_sdk_streams_large_workspace_backups_without_full_buffering(api_server):
    import io

    output = io.BytesIO()
    with AxiomClient(api_server) as client:
        written = client.download_workspace_backup_to("ws-1", output, chunk_size=4096)
    assert written == len(output.getvalue())
    assert output.getvalue().startswith(b"PK")

def test_python_sdk_uploads_and_downloads_workspace_backups(api_server):
    archive = b"PK\x03\x04uploaded"
    with AxiomClient(api_server) as client:
        downloaded = client.download_workspace_backup("ws-1")
        restored = client.restore_workspace_backup(archive, "Restored / test")
    assert downloaded.startswith(b"PK")
    assert restored["workspace"]["id"] == "ws-restored"
    request = _Handler.requests[-1]
    assert request["method"] == "POST"
    assert request["path"] == "/api/workspaces/backup/restore"
    assert request["query"] == {"name": ["Restored / test"]}
    assert {key.casefold(): value for key, value in request["headers"].items()}["content-type"] == "application/zip"
    assert request["body"] == archive


def test_python_sdk_stops_redirects_before_credentials_can_escape(api_server):
    _Handler.redirect_target = "https://example.invalid/collect"
    with AxiomClient(api_server, token="secret") as client:
        with pytest.raises(AxiomAPIError) as raised:
            client.request_json("GET", "/api/redirect")
    assert raised.value.status_code == 302
    assert len(_Handler.requests) == 1


def test_python_sdk_rejects_insecure_remote_origins_and_invalid_paths():
    with pytest.raises(AxiomConfigurationError, match="HTTPS"):
        AxiomClient("http://api.example.com")
    with pytest.raises(AxiomConfigurationError):
        AxiomClient("http://user:password@127.0.0.1:40507")
    with AxiomClient() as client:
        with pytest.raises(AxiomConfigurationError):
            client.request_json("GET", "https://example.invalid/api/health")
        with pytest.raises(AxiomConfigurationError):
            client.request_json("GET", "/api/../openapi.json")


def test_python_sdk_rejects_invalid_ports_credentials_and_duplicate_headers():
    for origin in (
        "http://127.0.0.1:70000",
        "https://user:password@example.com",
    ):
        with pytest.raises(AxiomConfigurationError):
            AxiomClient(origin)

    with pytest.raises(AxiomConfigurationError, match="control"):
        AxiomClient(token="secret\r\nX-Evil: true")
    with pytest.raises(AxiomConfigurationError, match="more than once"):
        AxiomClient(headers={"X-Client": "one", "x-client": "two"})
    with pytest.raises(AxiomConfigurationError, match="printable"):
        AxiomClient(workspace_id="workspace\nheader")


def test_python_sdk_closes_cleanly(api_server):
    client = AxiomClient(api_server)
    client.close()
    with pytest.raises(AxiomConfigurationError, match="closed"):
        client.health()


def test_python_sdk_models_no_implicit_retries_or_network_fallback(api_server):
    with AxiomClient(api_server) as client:
        with pytest.raises(AxiomAPIError) as raised:
            client.record_telemetry_sample()
    assert raised.value.status_code == 409
    request = _Handler.requests[-1]
    assert request["body"] == b""
    assert len(_Handler.requests) == 1

def test_python_sdk_prediction_jobs_support_start_history_status_resume_and_cancel(api_server):
    request = {
        "model_path": "models/tiny",
        "dataset_path": "data/eval.jsonl",
        "output_path": "data/predictions.jsonl",
        "max_new_tokens": 16,
        "device": "cpu",
    }
    job_id = "a" * 32
    with AxiomClient(api_server) as client:
        page = client.prediction_jobs(limit=12)
        started = client.start_prediction_job(request)
        shown = client.prediction_job(job_id)
        resumed = client.resume_prediction_job(job_id)
        cancelled = client.cancel_prediction_job(job_id)

    assert page["jobs"][0]["resumable"] is True
    assert started["job"]["status"] == "queued"
    assert shown["job"]["id"] == job_id
    assert resumed["job"]["resumption_count"] == 1
    assert resumed["scope"]["generation_resumed_after_restart"] is True
    assert cancelled["cancellation_accepted"] is True
    records = _Handler.requests
    assert [item["path"] for item in records] == [
        "/api/prediction-jobs",
        "/api/prediction-jobs",
        f"/api/prediction-jobs/{job_id}",
        f"/api/prediction-jobs/{job_id}/resume",
        f"/api/prediction-jobs/{job_id}/cancel",
    ]
    assert records[0]["query"] == {"limit": ["12"]}
    assert json.loads(records[1]["body"]) == request
    assert records[3]["body"] == b""
    assert records[4]["body"] == b""


def test_python_sdk_prediction_job_methods_reject_unsafe_ids_and_invalid_inputs():
    with AxiomClient() as client:
        with pytest.raises(AxiomConfigurationError, match="32 lowercase hexadecimal"):
            client.resume_prediction_job("../unsafe")
        with pytest.raises(AxiomConfigurationError, match="model_path"):
            client.start_prediction_job({"dataset_path": "data/eval.jsonl", "output_path": "data/out.jsonl"})
        with pytest.raises(AxiomConfigurationError, match="temperature"):
            client.start_prediction_job({
                "model_path": "models/tiny",
                "dataset_path": "data/eval.jsonl",
                "output_path": "data/out.jsonl",
                "temperature": float("nan"),
            })

        base = {
            "model_path": "models/tiny",
            "dataset_path": "data/eval.jsonl",
            "output_path": "data/out.jsonl",
        }
        with pytest.raises(AxiomConfigurationError, match="direct-child"):
            client.start_prediction_job({**base, "output_path": "data/nested/out.jsonl"})
        with pytest.raises(AxiomConfigurationError, match="different"):
            client.start_prediction_job({**base, "prediction_field": "prompt"})
        with pytest.raises(AxiomConfigurationError, match="JSON field name"):
            client.start_prediction_job({**base, "prompt_field": "not a field"})
        with pytest.raises(AxiomConfigurationError, match="Metal/MPS"):
            client.start_prediction_job({**base, "device": "mps", "precision": "bf16"})


