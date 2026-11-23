from __future__ import annotations

import io
import json
import os
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from axiom.cli.main import app

from axiom.core import supply_chain
from axiom.core.supply_chain import (
    SupplyChainError,
    audit_dependency_vulnerabilities,
    generate_spdx_sbom,
    write_report_new,
)


def distribution(name: str, version: str, requires: list[str] | None = None):
    return SimpleNamespace(
        metadata={"Name": name, "Summary": "An installed package"},
        version=version,
        requires=requires or [],
    )


def test_spdx_sbom_contains_package_urls_versions_and_installed_dependency_relationships():
    generated = datetime(2026, 11, 21, 0, 0, tzinfo=UTC)
    report = generate_spdx_sbom(
        distributions=[
            distribution("AXIOM_Core", "1.2.3", ["FastAPI>=0.100", "missing-extra>=1"]),
            distribution("fastapi", "0.115.0", ["starlette>=0.40"]),
            distribution("starlette", "0.41.0"),
        ],
        generated_at=generated,
    )

    assert report["spdxVersion"] == "SPDX-2.3"
    assert report["creationInfo"]["created"] == "2026-11-21T00:00:00Z"
    assert len(report["packages"]) == 3
    annotation = report["annotations"][0]["comment"]
    assert "content_sha256=" in annotation
    packages = {item["name"]: item for item in report["packages"]}
    assert packages["AXIOM_Core"]["externalRefs"][0]["referenceLocator"] == "pkg:pypi/axiom-core@1.2.3"
    assert packages["fastapi"]["versionInfo"] == "0.115.0"
    assert all(item["filesAnalyzed"] is False for item in packages.values())
    assert all(item["licenseDeclared"] == "NOASSERTION" for item in packages.values())
    links = {(item["spdxElementId"], item["relatedSpdxElement"], item["relationshipType"]) for item in report["relationships"]}
    assert (packages["AXIOM_Core"]["SPDXID"], packages["fastapi"]["SPDXID"], "DEPENDS_ON") in links
    assert (packages["fastapi"]["SPDXID"], packages["starlette"]["SPDXID"], "DEPENDS_ON") in links
    assert not any("missing-extra" in str(item) for item in report["relationships"])
    assert "network_access_performed=false" in annotation
    assert "host_paths_collected=false" in annotation


def test_spdx_sbom_validates_against_the_official_23_json_schema():
    import jsonschema

    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "docs/schemas/spdx-2.3.1.schema.json").read_text(encoding="utf-8"))
    report = generate_spdx_sbom(
        distributions=[
            distribution("axiom-core", "1.2.3", ["fastapi>=0.100"]),
            distribution("fastapi", "0.115.0"),
        ],
        generated_at=datetime(2026, 11, 22, 0, 0, tzinfo=UTC),
    )
    validator = jsonschema.Draft201909Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda item: tuple(str(part) for part in item.absolute_path))
    assert not errors, [error.message for error in errors]
    assert "axiom" not in report


def test_spdx_sbom_is_stable_for_the_same_packages_and_timestamp():
    packages = [distribution("fastapi", "0.115.0"), distribution("typer", "0.12.5")]
    timestamp = datetime(2026, 11, 21, 0, 0, tzinfo=UTC)
    assert generate_spdx_sbom(distributions=packages, generated_at=timestamp) == generate_spdx_sbom(
        distributions=packages, generated_at=timestamp
    )




def test_spdx_sbom_percent_encodes_purl_version_epochs():
    report = generate_spdx_sbom(distributions=[distribution("legacy-package", "1!2.3+local")])
    package = report["packages"][0]
    assert package["externalRefs"][0]["referenceLocator"] == "pkg:pypi/legacy-package@1%212.3+local"

def test_spdx_sbom_bounds_lazy_dependency_metadata_iterators():
    oversized = SimpleNamespace(
        metadata={"Name": "large-requirements"},
        version="1.0",
        requires=(f"dependency-{index}>=1" for index in range(4097)),
    )
    with pytest.raises(SupplyChainError, match="Dependency metadata"):
        generate_spdx_sbom(distributions=[oversized])


def test_spdx_sbom_bounds_aggregate_requirement_metadata(monkeypatch: pytest.MonkeyPatch):
    packages = [
        distribution("package-one", "1.0", requires=["dependency-a>=1", "dependency-b>=1"]),
        distribution("package-two", "1.0", requires=["dependency-c>=1"]),
    ]
    monkeypatch.setattr(supply_chain, "MAX_TOTAL_REQUIREMENTS", 2)
    with pytest.raises(SupplyChainError, match="requirement entries"):
        generate_spdx_sbom(distributions=packages)


def test_spdx_sbom_rejects_unbounded_or_invalid_installed_metadata():
    too_many = [distribution(f"pkg-{index}", "1.0") for index in range(supply_chain.MAX_DISTRIBUTIONS + 1)]
    with pytest.raises(SupplyChainError, match="exceeds"):
        generate_spdx_sbom(distributions=too_many)
    with pytest.raises(SupplyChainError, match="invalid"):
        generate_spdx_sbom(distributions=[distribution("../unsafe", "1.0")])


def test_audit_parser_bounds_total_finding_inventory():
    # pip-audit caps one dependency's vulnerability list separately. Exercise
    # the aggregate bound with many individually valid dependency records.
    payload = [
        {
            "name": f"example-{package_index}",
            "version": "1.0",
            "vulns": [
                {
                    "id": f"PYSEC-2026-{package_index}-{finding_index}",
                    "aliases": [],
                    "fix_versions": [],
                }
                for finding_index in range(1000)
            ],
        }
        for package_index in range((supply_chain.MAX_AUDIT_FINDINGS // 1000) + 1)
    ]
    with pytest.raises(SupplyChainError, match="finding inventory"):
        supply_chain._parse_pip_audit_output(
            payload,
            generated_at="2026-11-21T00:00:00Z",
            scanner_version="2.10.1",
            exit_code=1,
        )


def test_report_export_is_create_only_private_and_rejects_overwrite(tmp_path: Path):
    output = tmp_path / "sbom.spdx.json"
    written = write_report_new(output, '{"spdxVersion":"SPDX-2.3"}\n')
    assert written == output
    assert json.loads(output.read_text(encoding="utf-8"))["spdxVersion"] == "SPDX-2.3"
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(SupplyChainError, match="already exists"):
        write_report_new(output, "{}")


def test_report_export_rejects_symlinked_parent_and_existing_symlink(tmp_path: Path):
    destination = tmp_path / "outside"
    destination.mkdir()
    parent_link = tmp_path / "linked"
    parent_link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(SupplyChainError, match="symbolic links"):
        write_report_new(parent_link / "report.json", "{}")

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    with pytest.raises(SupplyChainError, match="already exists"):
        write_report_new(link, "{}")


def test_vulnerability_scan_requires_explicit_network_authorization():
    with pytest.raises(SupplyChainError, match="explicit network authorization"):
        audit_dependency_vulnerabilities(allow_network=False)


def test_security_api_exports_spdx_and_requires_strict_network_authorization(monkeypatch: pytest.MonkeyPatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from axiom.api.security import router as security_router

    sbom = {
        "spdxVersion": "SPDX-2.3",
        "SPDXID": "SPDXRef-DOCUMENT",
    }
    monkeypatch.setattr("axiom.api.security.generate_spdx_sbom", lambda: sbom)
    from axiom.core.supply_chain import audit_dependency_vulnerabilities as run_audit

    calls = []

    def fake_audit(*, allow_network: bool, timeout_seconds: int):
        calls.append((allow_network, timeout_seconds))
        if not allow_network:
            return run_audit(allow_network=allow_network, timeout_seconds=timeout_seconds)
        return {
            "schema_version": 1,
            "status": "unavailable",
            "network_accessed": False,
            "dependency_count": 0,
            "vulnerable_dependency_count": 0,
            "vulnerability_count": 0,
            "findings": [],
            "message": "No scanner available; no conclusion.",
        }

    monkeypatch.setattr("axiom.api.security.audit_dependency_vulnerabilities", fake_audit)
    application = FastAPI()
    application.include_router(security_router)
    client = TestClient(application)

    exported = client.get("/api/security/sbom")
    assert exported.status_code == 200
    assert exported.json() == sbom

    denied = client.post("/api/security/dependency-audit", json={"allow_network": False})
    assert denied.status_code == 422
    assert "explicit authorization" in denied.json()["detail"]
    assert calls == [(False, 90)]

    malformed = client.post("/api/security/dependency-audit", json={"allow_network": "true"})
    assert malformed.status_code == 422
    allowed = client.post("/api/security/dependency-audit", json={
        "allow_network": True,
        "timeout_seconds": 45,
    })
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "unavailable"
    assert calls == [(False, 90), (True, 45)]


def test_vulnerability_scan_reports_unavailable_scanner_without_claiming_clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: None)
    report = audit_dependency_vulnerabilities(allow_network=True)
    assert report["status"] == "unavailable"
    assert report["network_accessed"] is False
    assert report["vulnerability_count"] == 0
    assert "no vulnerability conclusion" in report["message"]


@pytest.mark.parametrize(
    ("exit_code", "findings"),
    [
        (0, [{"id": "PYSEC-2026-1"}]),
        (1, []),
    ],
)
def test_pip_audit_parser_rejects_exit_status_and_finding_mismatches(exit_code: int, findings: list[dict]):
    payload = [{"name": "example", "version": "1.0", "vulns": findings}]
    with pytest.raises(SupplyChainError, match="conflicts"):
        supply_chain._parse_pip_audit_output(
            payload,
            generated_at="2026-11-22T00:00:00Z",
            scanner_version="2.10.1",
            exit_code=exit_code,
        )


def test_pip_audit_parser_sanitizes_findings_and_deduplicates_records():
    payload = [
        {"name": "FastAPI", "version": "0.90.0", "vulns": [
            {"id": "PYSEC-2024-1", "aliases": ["CVE-2024-0001", "bad alias !"], "fix_versions": ["0.100.0"]},
            {"id": "PYSEC-2024-1", "aliases": ["CVE-2024-0001"], "fix_versions": ["0.100.0"]},
        ]},
    ]
    result = supply_chain._parse_pip_audit_output(
        payload, generated_at="2026-11-21T00:00:00Z", scanner_version="2.9.0", exit_code=1
    )
    assert result["status"] == "complete"
    assert result["dependency_count"] == 1
    assert result["vulnerable_dependency_count"] == 1
    assert result["vulnerability_count"] == 1
    assert result["findings"][0]["aliases"] == ["CVE-2024-0001"]
    assert result["findings"][0]["fix_versions"] == ["0.100.0"]
    assert result["network_accessed"] is True


def test_pip_audit_parser_accepts_a_successful_scan_with_no_findings():
    result = supply_chain._parse_pip_audit_output(
        [{"name": "example-package", "version": "1.2.3", "vulns": []}],
        generated_at="2026-11-23T00:00:00Z",
        scanner_version="2.10.1",
        exit_code=0,
    )
    assert result["status"] == "complete"
    assert result["network_accessed"] is True
    assert result["dependency_count"] == 1
    assert result["vulnerable_dependency_count"] == 0
    assert result["vulnerability_count"] == 0
    assert result["findings"] == []


def test_vulnerability_scan_invokes_scanner_only_after_authorization_and_drops_secret_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = [{"name": "typer", "version": "0.12.0", "vulns": [
        {"id": "PYSEC-2026-1", "aliases": ["CVE-2026-0001"], "fix_versions": ["0.13.0"]},
    ]}]
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(supply_chain.importlib.metadata, "version", lambda _name: "2.9.0")
    monkeypatch.setenv("PIP_INDEX_URL", "https://private.example/token")
    monkeypatch.setenv("AXIOM_API_TOKEN", "must-not-leak")
    calls = []

    class FakeProcess:
        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))
            self.stdout = io.BytesIO(json.dumps(payload).encode("utf-8"))
            self.stderr = io.BytesIO()
            self.returncode = 1

        def wait(self, timeout=None):
            assert timeout == 30
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(supply_chain.subprocess, "Popen", FakeProcess)
    monkeypatch.chdir(tmp_path)
    report = audit_dependency_vulnerabilities(allow_network=True, timeout_seconds=30)

    assert report["status"] == "complete"
    assert report["vulnerability_count"] == 1
    args, kwargs = calls[0]
    assert args[-4:] == ["--format", "json", "--progress-spinner", "off"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["cwd"] == supply_chain.sys.prefix
    assert "PIP_INDEX_URL" not in kwargs["env"]
    assert "AXIOM_API_TOKEN" not in kwargs["env"]
    assert kwargs["env"]["PIP_CONFIG_FILE"] == os.devnull


def test_vulnerability_scan_is_single_flight_and_does_not_spawn_concurrent_scanners():
    assert supply_chain._AUDIT_LOCK.acquire(blocking=False)
    try:
        report = audit_dependency_vulnerabilities(allow_network=True)
    finally:
        supply_chain._AUDIT_LOCK.release()
    assert report["status"] == "error"
    assert report["network_accessed"] is False
    assert "already active" in report["message"]


def test_vulnerability_scan_rejects_duplicate_json_object_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(supply_chain.importlib.metadata, "version", lambda _name: "2.10.1")

    class DuplicateJSONProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.BytesIO(b'{"dependencies":[],"dependencies":[]}')
            self.stderr = io.BytesIO()
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(supply_chain.subprocess, "Popen", DuplicateJSONProcess)
    with pytest.raises(SupplyChainError, match="ambiguous JSON"):
        audit_dependency_vulnerabilities(allow_network=True)


def test_vulnerability_scan_does_not_report_clean_on_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(supply_chain.importlib.metadata, "version", lambda _name: "2.9.0")

    class TimeoutProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.killed = False

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired(cmd="pip-audit", timeout=timeout)
            return -9

        def kill(self):
            self.killed = True

    monkeypatch.setattr(supply_chain.subprocess, "Popen", TimeoutProcess)
    result = audit_dependency_vulnerabilities(allow_network=True, timeout_seconds=1)
    assert result["status"] == "error"
    assert result["network_accessed"] is True
    assert "no complete vulnerability conclusion" in result["message"]


@pytest.mark.parametrize(
    ("stream_name", "limit_name"),
    [("stdout", "MAX_AUDIT_OUTPUT_BYTES"), ("stderr", "MAX_AUDIT_STDERR_BYTES")],
)
def test_vulnerability_scan_bounds_captured_streams(monkeypatch: pytest.MonkeyPatch, stream_name: str, limit_name: str):
    limit = 128
    monkeypatch.setattr(supply_chain, limit_name, limit)
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(supply_chain.importlib.metadata, "version", lambda _name: "2.9.0")

    class NoisyProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.BytesIO(b"x" * (limit + 1) if stream_name == "stdout" else b"{}")
            self.stderr = io.BytesIO(b"x" * (limit + 1) if stream_name == "stderr" else b"")
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(supply_chain.subprocess, "Popen", NoisyProcess)
    report = audit_dependency_vulnerabilities(allow_network=True, timeout_seconds=5)
    assert report["status"] == "error"
    assert report["network_accessed"] is True
    assert "exceeded" in report["message"]


def test_security_cli_exports_spdx_sbom_to_a_create_only_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = {
        "spdxVersion": "SPDX-2.3",
    }
    monkeypatch.setattr("axiom.cli.extended.generate_spdx_sbom", lambda: report)
    output = tmp_path / "sbom.spdx.json"
    result = CliRunner().invoke(app, ["security", "sbom", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    again = CliRunner().invoke(app, ["security", "sbom", "--output", str(output)])
    assert again.exit_code == 2
    assert output.read_text(encoding="utf-8") == json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_security_cli_requires_network_flag_for_vulnerability_scans():
    result = CliRunner().invoke(app, ["security", "vulnerabilities", "--json"])
    assert result.exit_code == 2
    assert "explicit network authorization" in result.output.lower()


def test_security_cli_passes_explicit_authorization_and_exports_audit_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = {
        "schema_version": 1,
        "status": "complete",
        "scanner": {"name": "pip-audit", "version": "2.9.0"},
        "dependency_count": 3,
        "vulnerable_dependency_count": 1,
        "vulnerability_count": 1,
        "findings": [{"id": "PYSEC-2026-1"}],
        "network_accessed": True,
    }
    calls = []
    def fake_audit(*, allow_network: bool, timeout_seconds: int):
        calls.append((allow_network, timeout_seconds))
        return report
    monkeypatch.setattr("axiom.cli.extended.audit_dependency_vulnerabilities", fake_audit)
    output = tmp_path / "dependency-audit.json"
    result = CliRunner().invoke(app, [
        "security", "vulnerabilities", "--allow-network", "--timeout", "45", "--output", str(output),
    ])
    assert result.exit_code == 0, result.output
    assert calls == [(True, 45)]
    assert json.loads(output.read_text(encoding="utf-8")) == report
