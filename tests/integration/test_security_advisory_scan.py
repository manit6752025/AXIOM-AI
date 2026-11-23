"""Real opt-in dependency advisory scan through AXIOM's security API implementation."""
from __future__ import annotations

import os

import pytest

from axiom.core.supply_chain import audit_dependency_vulnerabilities

pytestmark = pytest.mark.skipif(
    os.environ.get("AXIOM_RUN_REAL_SECURITY_SCAN") != "1",
    reason="The real advisory scan contacts configured public advisory services and runs only in its dedicated CI job.",
)


def test_axiom_runs_real_local_environment_advisory_scan_without_findings():
    report = audit_dependency_vulnerabilities(allow_network=True, timeout_seconds=300)

    assert report["schema_version"] == 1
    assert report["scanner"]["name"] == "pip-audit"
    assert report["scanner"]["version"]
    assert report["status"] == "complete", report["message"]
    assert report["network_accessed"] is True
    assert report["dependency_count"] > 0
    assert report["vulnerable_dependency_count"] == 0
    assert report["vulnerability_count"] == 0
    assert report["findings"] == []
    assert report["privacy"]["credentials_included"] is False
    assert report["privacy"]["environment_values_included"] is False
