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


def test_sdk_release_pipeline_is_manual_oidc_verified_and_installs_built_artifacts():
    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/sdk-release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    package = json.loads((root / "sdk/typescript/package.json").read_text(encoding="utf-8"))

    assert list(workflow["on"]) == ["workflow_dispatch"]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["package"]["options"] == [
        "both", "python", "typescript", "validate-only",
    ]
    assert package["repository"] == {
        "type": "git",
        "url": "https://github.com/manit6752025/AXIOM-AI",
    }

    jobs = workflow["jobs"]
    validate = jobs["validate-and-build"]
    validate_runs = [step.get("run", "") for step in validate["steps"]]
    joined_validate = "\\n".join(validate_runs)
    assert "python -m build sdk/python" in joined_validate
    assert "npm test --prefix sdk/typescript" in joined_validate
    assert "npm pack --prefix sdk/typescript" in joined_validate
    assert "axiom-sdk-wheel-check" in joined_validate
    assert "axiom-sdk-tarball-check" in joined_validate
    assert "npm install" in joined_validate
    assert any(
        step.get("uses") == "actions/upload-artifact@v7"
        and step.get("with", {}).get("name") == "axiom-sdk-release-artifacts"
        and step.get("with", {}).get("path") == "dist/"
        for step in validate["steps"]
    )
    assert "validate-only" in jobs["validate-and-build"]["steps"][5].get("if", "")
    assert any("validate-only" in step.get("if", "") for step in validate["steps"])

    python_publish = jobs["publish-python"]
    assert python_publish["environment"]["name"] == "pypi"
    assert python_publish["permissions"]["id-token"] == "write"
    assert any("pypa/gh-action-pypi-publish" in step.get("uses", "") for step in python_publish["steps"])

    npm_publish = jobs["publish-typescript"]
    assert npm_publish["environment"]["name"] == "npm"
    assert npm_publish["runs-on"] == "ubuntu-latest"
    assert npm_publish["permissions"]["id-token"] == "write"
    npm_runs = "\\n".join(step.get("run", "") for step in npm_publish["steps"])
    npm_setup = next(step for step in npm_publish["steps"] if step.get("uses") == "actions/setup-node@v7")
    assert npm_setup["with"]["node-version"] == "22"
    assert npm_setup["with"]["registry-url"] == "https://registry.npmjs.org"
    assert npm_setup["with"]["package-manager-cache"] == "false"
    assert "npm@11.15.0" in npm_runs
    assert "npm publish dist/typescript/*.tgz --provenance --access public" in npm_runs
    assert "NPM_TOKEN" not in json.dumps(npm_publish)

    python_verify = jobs["verify-python-install"]
    assert python_verify["needs"] == "publish-python"
    assert "axiom-ai-sdk==1.0.0" in "\\n".join(step.get("run", "") for step in python_verify["steps"])
    npm_verify = jobs["verify-typescript-install"]
    assert npm_verify["needs"] == "publish-typescript"
    assert "@axiom-ai/sdk@1.0.0" in "\\n".join(step.get("run", "") for step in npm_verify["steps"])
