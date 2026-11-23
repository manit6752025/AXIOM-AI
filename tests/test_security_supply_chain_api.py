from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import security as security_api


def test_security_supply_chain_api_returns_spdx_sbom(tmp_path, monkeypatch):
    report = {"spdxVersion": "SPDX-2.3", "axiom": {"package_count": 1}}
    monkeypatch.setattr(security_api, "generate_spdx_sbom", lambda: report)
    app = FastAPI()
    app.include_router(security_api.router)
    response = TestClient(app).get("/api/security/sbom")
    assert response.status_code == 200
    assert response.json() == report


def test_security_supply_chain_api_requires_explicit_network_authorization():
    app = FastAPI()
    app.include_router(security_api.router)
    response = TestClient(app).post("/api/security/dependency-audit", json={"allow_network": False})
    assert response.status_code == 422


def test_security_supply_chain_api_runs_only_authorized_bounded_dependency_scan(monkeypatch):
    report = {"schema_version": 1, "status": "complete", "vulnerability_count": 0, "network_accessed": True}
    calls = []
    def fake_audit(*, allow_network: bool, timeout_seconds: int):
        calls.append((allow_network, timeout_seconds))
        return report
    monkeypatch.setattr(security_api, "audit_dependency_vulnerabilities", fake_audit)
    app = FastAPI()
    app.include_router(security_api.router)
    client = TestClient(app)
    success = client.post("/api/security/dependency-audit", json={"allow_network": True, "timeout_seconds": 25})
    extra = client.post("/api/security/dependency-audit", json={"allow_network": True, "timeout_seconds": 25, "token": "secret"})
    out_of_range = client.post("/api/security/dependency-audit", json={"allow_network": True, "timeout_seconds": 301})
    assert success.status_code == 200
    assert success.json() == report
    assert extra.status_code == 422
    assert out_of_range.status_code == 422
    assert calls == [(True, 25)]
