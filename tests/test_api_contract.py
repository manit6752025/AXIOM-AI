from fastapi.testclient import TestClient

from axiom.api.server import API_CONTRACT_VERSION, create_app
from axiom.version import __version__


def test_api_contract_exposes_versioned_openapi_and_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        metadata = client.get("/api/meta")
        assert metadata.status_code == 200
        assert metadata.json()["api_contract_version"] == API_CONTRACT_VERSION == 1
        assert metadata.json()["application_version"] == __version__
        assert metadata.json()["openapi_url"] == "/api/openapi.json"

        contract = client.get("/api/openapi.json")
        assert contract.status_code == 200
        assert contract.json()["info"]["version"] == __version__
        assert "/api/workspace/experiments" in contract.json()["paths"]
        assert client.get("/openapi.json").status_code == 404

        invalid_id = client.get("/api/health", headers={"X-Request-ID": "not a valid id"})
        assert invalid_id.status_code == 200
        assert invalid_id.headers["X-Request-ID"] != "not a valid id"
        assert len(invalid_id.headers["X-Request-ID"]) == 32

        missing = client.get("/api/evaluation/runs/missing", headers={"X-Request-ID": "trace_contract_test"})
        assert missing.status_code == 404
        assert missing.headers["X-Request-ID"] == "trace_contract_test"
        assert missing.json()["request_id"] == "trace_contract_test"
        assert missing.json()["error"]["code"] == "http_error"
