from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.api.server import API_CONTRACT_VERSION, create_app
from scripts.verify_sdk_contract import SDKContractError, verify_sdk_contract


def test_sdk_operations_match_live_openapi_contract():
    report = verify_sdk_contract()
    assert report["status"] == "verified"
    assert report["api_contract_version"] == API_CONTRACT_VERSION == 1
    assert report["operation_count"] >= 15


def test_sdk_operation_manifest_paths_and_methods_exist_in_openapi():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "sdk/openapi/axiom-v1-operations.json").read_text(encoding="utf-8"))
    paths = create_app().openapi()["paths"]
    for operation in manifest["operations"]:
        assert operation["path"] in paths
        assert operation["method"].lower() in paths[operation["path"]]


def test_sdk_contract_rejects_manifest_drift(monkeypatch, tmp_path):
    import scripts.verify_sdk_contract as verifier

    original = verifier.MANIFEST_PATH.read_text(encoding="utf-8")
    malformed = json.loads(original)
    malformed["operations"][0]["path"] = "/api/not-a-real-route"
    temporary = tmp_path / "drifted-operations.json"
    temporary.write_text(json.dumps(malformed), encoding="utf-8")
    monkeypatch.setattr(verifier, "MANIFEST_PATH", temporary)
    with pytest.raises(SDKContractError, match="OpenAPI"):
        verifier.verify_sdk_contract()
