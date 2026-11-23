from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import axiom.api.server as server_module
import axiom.api.server as server_module
from axiom.api.server import _required_api_scope, create_app


READ_TOKEN = "read_" + "r" * 40
WRITE_TOKEN = "write_" + "w" * 40


def test_scope_classifier_defaults_mutations_to_write() -> None:
    assert _required_api_scope("GET", "/api/workflows") == "read"
    assert _required_api_scope("POST", "/api/workflows/validate") == "read"
    assert _required_api_scope("POST", "/api/workflows/workflow_123/dry-run") == "read"
    assert _required_api_scope("POST", "/api/optimizer/plan") == "read"
    assert _required_api_scope("POST", "/api/workflows") == "write"
    assert _required_api_scope("POST", "/api/fine-tuning/jobs") == "write"
    assert _required_api_scope("POST", "/api/engine/generate") == "write"
    assert _required_api_scope("POST", "/v1/chat/completions") == "write"
    assert _required_api_scope("PUT", "/api/workflows/workflow_123") == "write"
    assert _required_api_scope("DELETE", "/api/workspace/models/model_123") == "write"


def test_api_scope_configuration_requires_two_distinct_strong_tokens(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_API_READ_TOKEN", READ_TOKEN)
    monkeypatch.delenv("AXIOM_API_WRITE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="must be configured together"):
        create_app()

    monkeypatch.setenv("AXIOM_API_WRITE_TOKEN", "short")
    with pytest.raises(RuntimeError, match="32"):
        create_app()

    monkeypatch.setenv("AXIOM_API_WRITE_TOKEN", READ_TOKEN)
    with pytest.raises(RuntimeError, match="must be different"):
        create_app()


def test_read_token_can_inspect_validate_and_dry_run_but_not_mutate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIOM_API_READ_TOKEN", READ_TOKEN)
    monkeypatch.setenv("AXIOM_API_WRITE_TOKEN", WRITE_TOKEN)

    read_headers = {"Authorization": f"Bearer {READ_TOKEN}"}
    write_headers = {"Authorization": f"Bearer {WRITE_TOKEN}"}
    workflow_payload = {
        "name": "Read-only plan",
        "nodes": [
            {"id": "dataset", "type": "dataset.profile"},
            {"id": "plan", "type": "training.plan"},
        ],
        "edges": [{"source": "dataset", "target": "plan"}],
    }

    with TestClient(create_app()) as client:
        missing = client.get(
            "/api/workflows",
            headers={"X-Request-ID": "scope_missing_01"},
        )
        assert missing.status_code == 401
        assert missing.headers["X-Request-ID"] == "scope_missing_01"
        assert missing.json()["error"]["code"] == "authentication_required"
        assert missing.json()["request_id"] == "scope_missing_01"

        assert client.get("/api/health").status_code == 200
        assert client.get("/api/workflows", headers=read_headers).status_code == 200
        security_audit = client.get("/api/security/audit", headers=read_headers)
        assert security_audit.status_code == 200
        audit_payload = security_audit.json()
        assert audit_payload["schema_version"] == 1
        assert READ_TOKEN not in security_audit.text
        assert WRITE_TOKEN not in security_audit.text

        validation = client.post(
            "/api/workflows/validate",
            json=workflow_payload,
            headers={**read_headers, "X-Request-ID": "scope_validate_01"},
        )
        assert validation.status_code == 200
        assert validation.json()["valid"] is True

        created = client.post("/api/workflows", json=workflow_payload, headers=write_headers)
        assert created.status_code == 201
        workflow_id = created.json()["workflow"]["id"]

        preview = client.post(
            f"/api/workflows/{workflow_id}/dry-run",
            json={"parameter_billions": 7, "method": "auto", "seed": 42},
            headers=read_headers,
        )
        assert preview.status_code == 200
        assert preview.json()["dry_run"]["status"] == "planning_only"

        denied_create = client.post(
            "/api/workflows",
            json={**workflow_payload, "name": "Must not be saved"},
            headers={**read_headers, "X-Request-ID": "scope_write_denied"},
        )
        assert denied_create.status_code == 403
        assert denied_create.headers["X-Request-ID"] == "scope_write_denied"
        assert denied_create.json()["error"]["code"] == "insufficient_scope"
        assert denied_create.json()["detail"] == {"required_scope": "write"}

        denied_job = client.post(
            "/api/fine-tuning/jobs",
            json={},
            headers=read_headers,
        )
        assert denied_job.status_code == 403

        denied_openai_completion = client.post(
            "/v1/chat/completions",
            json={},
            headers=read_headers,
        )
        assert denied_openai_completion.status_code == 403


def test_same_origin_control_center_remains_available_with_scopes_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIOM_API_READ_TOKEN", READ_TOKEN)
    monkeypatch.setenv("AXIOM_API_WRITE_TOKEN", WRITE_TOKEN)

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/workflows",
            headers={
                "Origin": "http://testserver",
                "Sec-Fetch-Site": "same-origin",
            },
        )
        assert response.status_code == 200


def test_ui_build_fingerprint_is_public_and_not_cacheable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    webui = tmp_path / "axiom" / "webui"
    webui.mkdir(parents=True)
    (webui / "index.html").write_text("<html>AXIOM</html>", encoding="utf-8")
    monkeypatch.setattr(server_module, "__file__", str(tmp_path / "axiom" / "api" / "server.py"))
    with TestClient(create_app()) as client:
        response = client.get("/api/ui/build")
    assert response.status_code == 200, response.text
    assert len(response.json()["fingerprint"]) == 64
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"


def test_ui_build_fingerprint_and_app_shell_are_safe_to_revalidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    webui = tmp_path / "axiom" / "webui"
    webui.mkdir(parents=True)
    (webui / "index.html").write_text("<html>AXIOM</html>", encoding="utf-8")
    monkeypatch.setattr(server_module, "__file__", str(tmp_path / "axiom" / "api" / "server.py"))
    with TestClient(create_app()) as client:
        fingerprint = client.get("/api/ui/build")
        app_shell = client.get("/ui/")
    assert fingerprint.status_code == 200, fingerprint.text
    assert len(fingerprint.json()["fingerprint"]) == 64
    assert fingerprint.headers["cache-control"] == "no-store, max-age=0"
    assert fingerprint.headers["pragma"] == "no-cache"
    assert app_shell.status_code == 200, app_shell.text
    assert app_shell.headers["cache-control"] == "no-cache, max-age=0"
