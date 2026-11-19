from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "axiom_compatibility_check", ROOT / "scripts" / "check_compatibility.py"
)
assert SPEC and SPEC.loader
compatibility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compatibility)


def test_supported_python_node_and_cross_platform_target_pass() -> None:
    report = compatibility.assess_compatibility((3, 11, 9), "Linux", "v22.12.0", require_node=True)
    assert report["status"] == "passed"
    assert report["checks"]["python"]["status"] == "pass"
    assert report["checks"]["node"]["status"] == "pass"
    assert report["checks"]["operating_system"]["status"] == "pass"


def test_python_below_project_minimum_fails() -> None:
    report = compatibility.assess_compatibility((3, 10, 14), "Windows", "v22.12.0", require_node=True)
    assert report["status"] == "failed"
    assert report["checks"]["python"]["status"] == "fail"


def test_node_versions_match_frontend_engine_requirements() -> None:
    assert compatibility.node_version_supported("v20.19.0")
    assert compatibility.node_version_supported("22.12.0")
    assert compatibility.node_version_supported("24.0.0")
    assert not compatibility.node_version_supported("20.18.1")
    assert not compatibility.node_version_supported("v22.11.9")
    assert not compatibility.node_version_supported("v21.7.0")
    assert not compatibility.node_version_supported("not-a-version")


def test_missing_node_is_skipped_for_python_only_install_and_fails_when_required() -> None:
    local = compatibility.assess_compatibility((3, 12, 1), "Darwin", None)
    assert local["status"] == "passed"
    assert local["checks"]["node"]["status"] == "skipped"

    release = compatibility.assess_compatibility((3, 12, 1), "Darwin", None, require_node=True)
    assert release["status"] == "failed"
    assert release["checks"]["node"]["status"] == "fail"


def test_unknown_os_is_reported_as_unverified_without_claiming_support() -> None:
    report = compatibility.assess_compatibility((3, 11, 8), "FreeBSD", None)
    assert report["status"] == "warning"
    assert report["checks"]["operating_system"]["status"] == "warning"
    assert "FreeBSD" not in report["checks"]["operating_system"]["tested_targets"]


def test_unusable_node_is_skipped_for_python_only_checks_but_fails_release_checks() -> None:
    local = compatibility.assess_compatibility((3, 11, 8), "Windows", "unavailable")
    assert local["status"] == "passed"
    assert local["checks"]["node"]["status"] == "skipped"

    release = compatibility.assess_compatibility((3, 11, 8), "Windows", "unavailable", require_node=True)
    assert release["status"] == "failed"
    assert release["checks"]["node"]["status"] == "fail"
