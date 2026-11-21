from __future__ import annotations

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
    assert report["axiom"]["package_count"] == 3
    packages = {item["name"]: item for item in report["packages"]}
    assert packages["AXIOM_Core"]["externalRefs"][0]["referenceLocator"] == "pkg:pypi/axiom-core@1.2.3"
    assert packages["fastapi"]["versionInfo"] == "0.115.0"
    assert all(item["filesAnalyzed"] is False for item in packages.values())
    assert all(item["licenseDeclared"] == "NOASSERTION" for item in packages.values())
    links = {(item["spdxElementId"], item["relatedSpdxElement"], item["relationshipType"]) for item in report["relationships"]}
    assert (packages["AXIOM_Core"]["SPDXID"], packages["fastapi"]["SPDXID"], "DEPENDS_ON") in links
    assert (packages["fastapi"]["SPDXID"], packages["starlette"]["SPDXID"], "DEPENDS_ON") in links
    assert not any("missing-extra" in str(item) for item in report["relationships"])
    assert report["axiom"]["network_access_performed"] is False
    assert report["axiom"]["host_paths_collected"] is False


def test_spdx_sbom_is_stable_for_the_same_packages_and_timestamp():
    packages = [distribution("fastapi", "0.115.0"), distribution("typer", "0.12.5")]
    timestamp = datetime(2026, 11, 21, 0, 0, tzinfo=UTC)
    assert generate_spdx_sbom(distributions=packages, generated_at=timestamp) == generate_spdx_sbom(
        distributions=packages, generated_at=timestamp
    )


def test_spdx_sbom_rejects_unbounded_or_invalid_installed_metadata():
    too_many = [distribution(f"pkg-{index}", "1.0") for index in range(supply_chain.MAX_DISTRIBUTIONS + 1)]
    with pytest.raises(SupplyChainError, match="exceeds"):
        generate_spdx_sbom(distributions=too_many)
    with pytest.raises(SupplyChainError, match="invalid"):
        generate_spdx_sbom(distributions=[distribution("../unsafe", "1.0")])


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


def test_vulnerability_scan_reports_unavailable_scanner_without_claiming_clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: None)
    report = audit_dependency_vulnerabilities(allow_network=True)
    assert report["status"] == "unavailable"
    assert report["network_accessed"] is False
    assert report["vulnerability_count"] == 0
    assert "no vulnerability conclusion" in report["message"]


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


def test_vulnerability_scan_invokes_scanner_only_after_authorization_and_drops_secret_environment(monkeypatch: pytest.MonkeyPatch):
    payload = [{"name": "typer", "version": "0.12.0", "vulns": [
        {"id": "PYSEC-2026-1", "aliases": ["CVE-2026-0001"], "fix_versions": ["0.13.0"]},
    ]}]
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(supply_chain.importlib.metadata, "version", lambda _name: "2.9.0")
    monkeypatch.setenv("PIP_INDEX_URL", "https://private.example/token")
    monkeypatch.setenv("AXIOM_API_TOKEN", "must-not-leak")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        kwargs["stdout"].write(json.dumps(payload).encode("utf-8"))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(supply_chain.subprocess, "run", fake_run)
    report = audit_dependency_vulnerabilities(allow_network=True, timeout_seconds=30)

    assert report["status"] == "complete"
    assert report["vulnerability_count"] == 1
    args, kwargs = calls[0]
    assert args[-4:] == ["--format", "json", "--progress-spinner", "off"]
    assert kwargs["timeout"] == 30
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "PIP_INDEX_URL" not in kwargs["env"]
    assert "AXIOM_API_TOKEN" not in kwargs["env"]


def test_vulnerability_scan_does_not_report_clean_on_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(supply_chain.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(supply_chain.importlib.metadata, "version", lambda _name: "2.9.0")
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="pip-audit", timeout=1)
    monkeypatch.setattr(supply_chain.subprocess, "run", timeout)
    result = audit_dependency_vulnerabilities(allow_network=True, timeout_seconds=1)
    assert result["status"] == "error"
    assert result["network_accessed"] is True
    assert "no complete vulnerability conclusion" in result["message"]


def test_security_cli_exports_spdx_sbom_to_a_create_only_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = {
        "spdxVersion": "SPDX-2.3",
        "axiom": {"package_count": 2},
    }
    monkeypatch.setattr("axiom.cli.extended.generate_spdx_sbom", lambda: report)
    output = tmp_path / "sbom.spdx.json"
    result = CliRunner().invoke(app, ["security", "sbom", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    again = CliRunner().invoke(app, ["security", "sbom", "--output", str(output)])
    assert again.exit_code == 2
    assert output.read_text(encoding="utf-8") == json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\\n"


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
