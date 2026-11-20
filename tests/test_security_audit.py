from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api.security import router as security_router
from axiom.core.security_audit import audit_security_posture
from axiom.core.tool_permissions import ToolPermissionStore

READ_TOKEN = "read_" + "r" * 40
WRITE_TOKEN = "write_" + "w" * 40


def _project(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = path / "axiom.yaml"
    config.write_text("project:\\n  name: secure-test\\n", encoding="utf-8")
    if os.name == "posix":
        config.chmod(0o600)
    return path


def _check(report: dict, check_id: str) -> dict:
    return next(item for item in report["checks"] if item["id"] == check_id)


def test_audit_reports_missing_api_scope_tokens_without_exposing_values(tmp_path: Path):
    secret = "NEVER-RETURN-THIS-API-SECRET"
    report = audit_security_posture(
        _project(tmp_path / "project"),
        environ={"AXIOM_API_READ_TOKEN": secret},
    )

    assert _check(report, "security.api.scopes")["status"] == "fail"
    assert secret not in json.dumps(report)


def test_audit_accepts_distinct_tokens_and_reports_only_presence(tmp_path: Path):
    env = {
        "AXIOM_API_READ_TOKEN": READ_TOKEN,
        "AXIOM_API_WRITE_TOKEN": WRITE_TOKEN,
        "AXIOM_OPENAI_API_TOKEN": "OPENAI_SECRET_SENTINEL",
    }
    report = audit_security_posture(_project(tmp_path / "project"), environ=env)

    assert report["summary"]["api_scope_tokens_enabled"] is True
    assert _check(report, "security.api.scopes")["status"] == "pass"
    assert _check(report, "security.api.openai_token")["status"] == "pass"
    serialized = json.dumps(report)
    for secret in (*env.values(), "AXIOM_API_READ_TOKEN"):
        assert secret not in serialized


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "http://public.example.com",
        "https://trusted.example.com/path",
        "https://user:password@example.com",
    ],
)
def test_audit_rejects_unsafe_or_non_origin_cors_values(tmp_path: Path, origin: str):
    report = audit_security_posture(
        _project(tmp_path / "project"),
        environ={"AXIOM_CORS_ORIGINS": origin},
    )
    assert _check(report, "security.cors.origins")["status"] == "fail"


def test_audit_warns_for_remote_https_origin_without_scope_auth(tmp_path: Path):
    report = audit_security_posture(
        _project(tmp_path / "project"),
        environ={"AXIOM_CORS_ORIGINS": "https://trusted.example.com"},
    )
    check = _check(report, "security.cors.origins")
    assert check["status"] == "warn"
    assert "CORS is not an authentication" in check["detail"]


def test_audit_flags_unauthenticated_non_loopback_daemon_profile(tmp_path: Path):
    report = audit_security_posture(
        _project(tmp_path / "project"),
        environ={},
        daemon_host="0.0.0.0",
        allow_network=True,
    )
    check = _check(report, "security.network.daemon_bind")
    assert check["status"] == "fail"
    assert "unauthenticated HTTP" in check["detail"]


def test_audit_checks_mcp_policy_and_reports_explicit_grants(tmp_path: Path):
    root = _project(tmp_path / "project")
    store = ToolPermissionStore(root / ".axiom")
    store.save_profile("review", ["axiom_info"])
    store.set_active("review")

    report = audit_security_posture(root, environ={})
    check = _check(report, "security.mcp.default_deny")
    assert check["status"] == "warn"
    assert "1 explicit tool grant" in check["detail"]


def test_audit_flags_group_writable_mcp_metadata_directory(tmp_path: Path):
    if os.name != "posix":
        pytest.skip("POSIX permission bits are not available on this platform")
    root = _project(tmp_path / "project")
    metadata = root / ".axiom"
    metadata.mkdir()
    metadata.chmod(0o770)

    report = audit_security_posture(root, environ={})
    assert _check(report, "security.mcp.policy_permissions")["status"] == "fail"


def test_security_audit_api_returns_redacted_read_only_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AXIOM_API_READ_TOKEN", raising=False)
    monkeypatch.delenv("AXIOM_API_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("AXIOM_OPENAI_API_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(security_router)

    response = TestClient(app).get("/api/security/audit")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["workspace"]["name"] == tmp_path.name
    assert any(item["id"] == "security.network.daemon_bind" for item in payload["checks"])
    assert not (tmp_path / ".axiom").exists()
