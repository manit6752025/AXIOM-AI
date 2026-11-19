#!/usr/bin/env python3
"""Check the Python, Node, and operating-system versions AXIOM currently validates.

This dependency-free check is suitable for CI and for local diagnosis. It does
not install packages, contact a network, or modify the environment.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from typing import Sequence

MIN_PYTHON = (3, 11, 0)
SUPPORTED_OPERATING_SYSTEMS = {"Linux", "Windows", "Darwin"}


def parse_node_version(value: str | None) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?\s*", value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def node_version_supported(value: str | None) -> bool:
    version = parse_node_version(value)
    if version is None:
        return False
    major, minor, patch = version
    if major == 20:
        return (minor, patch) >= (19, 0)
    if major == 22:
        return minor >= 12
    return major > 22


def assess_compatibility(
    python_version: Sequence[int],
    operating_system: str,
    node_version: str | None,
    *,
    require_node: bool = False,
) -> dict[str, object]:
    """Evaluate supplied runtime facts, isolated for tests and callers."""
    python_tuple = tuple(int(part) for part in python_version[:3])
    if len(python_tuple) < 3:
        python_tuple = (*python_tuple, *(0 for _ in range(3 - len(python_tuple))))
    python_ok = python_tuple >= MIN_PYTHON
    os_ok = operating_system in SUPPORTED_OPERATING_SYSTEMS
    node_unavailable = node_version is None or node_version == "unavailable"
    if node_unavailable:
        node_status = "fail" if require_node else "skipped"
        node_text = "not installed or not usable on PATH"
    else:
        node_supported = node_version_supported(node_version)
        node_status = "pass" if node_supported else "fail"
        node_text = node_version

    checks = {
        "python": {
            "status": "pass" if python_ok else "fail",
            "actual": ".".join(str(part) for part in python_tuple),
            "requirement": ">=3.11.0",
        },
        "node": {
            "status": node_status,
            "actual": node_text,
            "requirement": "20.19.0+ or 22.12.0+ (including later majors)",
            "required_for_this_check": require_node,
        },
        "operating_system": {
            "status": "pass" if os_ok else "warning",
            "actual": operating_system,
            "tested_targets": sorted(SUPPORTED_OPERATING_SYSTEMS),
        },
    }
    statuses = {str(check["status"]) for check in checks.values()}
    overall = "failed" if "fail" in statuses else "warning" if "warning" in statuses else "passed"
    return {
        "schema_version": 1,
        "status": overall,
        "checks": checks,
        "scope_note": (
            "Compatibility preflight only; it does not prove GPU-driver compatibility or model-specific runtime support."
        ),
    }


def _probe_node() -> str | None:
    executable = shutil.which("node")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a versioned JSON report.")
    parser.add_argument("--require-node", action="store_true", help="Fail when a compatible Node.js is unavailable.")
    args = parser.parse_args(argv)
    report = assess_compatibility(
        sys.version_info[:3],
        platform.system(),
        _probe_node(),
        require_node=args.require_node,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"AXIOM runtime compatibility: {report['status'].upper()}")
        for name, check in report["checks"].items():
            print(f"{name}: {check['status']} ({check['actual']})")
        print(report["scope_note"])
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
