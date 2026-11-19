#!/usr/bin/env python3
"""Dependency-free static acceptance checks for an AXIOM checkout.

This complements (rather than replaces) pytest, ESLint, the production build, and
Playwright tests. It never imports the application or contacts external services.
"""
from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROADMAP_PATTERN = re.compile(r"^- \[(R\d{3}) · ([^\]]+)\]", re.MULTILINE)
VALID_STATUSES = {
    "Implemented; tests passing",
    "Planned",
    "In progress",
    "In progress; validation pending",
}


def roadmap_issues(markdown: str) -> list[str]:
    entries = ROADMAP_PATTERN.findall(markdown)
    ids = [item[0] for item in entries]
    expected = {f"R{i:03d}" for i in range(1, 121)}
    issues: list[str] = []
    missing = sorted(expected - set(ids))
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    unexpected = sorted(set(ids) - expected)
    unknown_statuses = sorted({status for _, status in entries if status not in VALID_STATUSES})
    if len(entries) != 120:
        issues.append(f"Expected 120 roadmap entries; found {len(entries)}.")
    if missing:
        issues.append("Missing roadmap IDs: " + ", ".join(missing))
    if duplicates:
        issues.append("Duplicate roadmap IDs: " + ", ".join(duplicates))
    if unexpected:
        issues.append("Unexpected roadmap IDs: " + ", ".join(unexpected))
    if unknown_statuses:
        issues.append("Unknown roadmap statuses: " + ", ".join(unknown_statuses))
    return issues


def page_registration_issues(app_source: str, page_id: str) -> list[str]:
    """Check that a page is represented in the Page union, nav, metadata and switch."""
    checks = {
        "Page union": bool(re.search(rf'^\s*\| "{re.escape(page_id)}"\s*$', app_source, re.MULTILINE)),
        "navigation": bool(re.search(rf'\bid:\s*"{re.escape(page_id)}"', app_source)),
        "page metadata": bool(re.search(rf'^\s*(?:"{re.escape(page_id)}"|{re.escape(page_id)}):\s*\{{', app_source, re.MULTILINE)),
        "render switch": bool(re.search(rf'case "{re.escape(page_id)}":', app_source)),
    }
    return [f"GUI page '{page_id}' is missing from {where}." for where, present in checks.items() if not present]


def module_catalog_issues(app_source: str, page_ids: tuple[str, ...] = ("api-playground", "agent-profiles")) -> list[str]:
    marker = "function ModulesPage"
    start = app_source.find(marker)
    if start < 0:
        return ["GUI Module Explorer function is missing."]
    catalog = app_source[start:]
    return [
        f"Module Explorer is missing an entry for '{page_id}'."
        for page_id in page_ids
        if f'id: "{page_id}"' not in catalog
    ]


def inspect(root: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    required = [
        "pyproject.toml", "axiom/version.py", "ROADMAP.md", "src/App.tsx",
        "src/api/client.ts", "src/components/ApiPlayground.tsx", "src/components/RuntimeConnectorPanel.tsx", "src/components/AgentProfiles.tsx", "src/styles.css",
        "axiom/api/server.py", "axiom/api/diagnostics.py", "axiom/api/runtime_discovery.py", "axiom/api/agent_profiles.py", "axiom/core/runtime_discovery.py", "axiom/core/agent_profiles.py", "axiom/cli/agent_profiles.py", "scripts/release_smoke.py",
        "tests/test_release_smoke.py", "e2e/helpers/auth.ts", "e2e/modules.spec.ts", "e2e/workflows.spec.ts", "e2e/evaluation.spec.ts",
        "tests/test_runtime_discovery.py", "tests/test_runtime_discovery_api.py", "tests/test_agent_profiles.py", "tests/test_agent_profiles_api.py", "tests/test_agent_profiles_cli.py", "e2e/api-playground.spec.ts", "e2e/runtime-connectors.spec.ts", "e2e/agent-profiles.spec.ts",
        "docs/Runtime-Discovery.md", "docs/Agent-Profiles.md", "docs/Api-Playground.md", ".github/workflows/quality-gates.yml",
    ]
    for relative in required:
        if not (root / relative).is_file():
            failures.append(f"Missing required file: {relative}")
    if failures:
        return {"schema_version": 1, "status": "failed", "failures": failures, "warnings": warnings}

    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project_version = str(project["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        failures.append(f"Could not read project metadata: {type(exc).__name__}")
        project_version = "unknown"

    version_source = (root / "axiom/version.py").read_text(encoding="utf-8")
    version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', version_source, re.MULTILINE)
    runtime_version = version_match.group(1) if version_match else "unknown"
    if project_version != runtime_version:
        failures.append(f"Python package version ({project_version}) differs from runtime version ({runtime_version}).")

    roadmap = (root / "ROADMAP.md").read_text(encoding="utf-8")
    failures.extend(roadmap_issues(roadmap))

    app_source = (root / "src/App.tsx").read_text(encoding="utf-8")
    failures.extend(page_registration_issues(app_source, "api-playground"))
    failures.extend(page_registration_issues(app_source, "agent-profiles"))
    failures.extend(module_catalog_issues(app_source))
    component = (root / "src/components/ApiPlayground.tsx").read_text(encoding="utf-8")
    client = (root / "src/api/client.ts").read_text(encoding="utf-8")
    for path in (
        "/api/health", "/api/meta", "/api/workspace/runtime",
        "/api/workspace/models?offset=0&limit=5", "/api/workspace/datasets",
        "/api/workspace/events", "/api/diagnostics/workspace",
        "/api/diagnostics/roadmap", "/api/diagnostics/runtime-connectors",
    ):
        if path not in client:
            failures.append(f"API playground allowlist is missing {path}.")
    if "playgroundRead" not in component or "GET only" not in component:
        failures.append("API playground must be wired to the typed GET-only client and disclose the safe boundary.")
    if "15_000" not in client:
        failures.append("The shared API client timeout is missing or has changed; review timeout policy.")
    styles_source = (root / "src/styles.css").read_text(encoding="utf-8")
    if "api-playground-json" not in styles_source:
        failures.append("API playground response styling is missing.")
    if "runtime-connector-row" not in styles_source:
        failures.append("Runtime connector panel styling is missing.")
    if "runtimeConnectors" not in client or "RuntimeConnectorPanel" not in app_source:
        failures.append("Runtime connector discovery is not wired through the client and GUI.")
    if not all(token in client for token in ("agentProfiles", "createAgentProfile", "updateAgentProfile", "deleteAgentProfile")):
        failures.append("Agent profile operations are missing from the typed API client.")
    if "AgentProfiles" not in app_source or "register_agent_profiles(app)" not in (root / "axiom/cli/main.py").read_text(encoding="utf-8"):
        failures.append("Agent profiles are not wired to both the GUI and CLI entry point.")
    server_source = (root / "axiom/api/server.py").read_text(encoding="utf-8")
    if "agent_profiles_router" not in server_source:
        failures.append("Agent profile API router is not registered.")
    runtime_api = (root / "axiom/api/runtime_discovery.py").read_text(encoding="utf-8")
    if '@router.get("/runtime-connectors")' not in runtime_api or "discover_runtime_connectors" not in runtime_api:
        failures.append("Runtime connector discovery endpoint is not registered.")
    if "runtime_discovery_router" not in server_source:
        failures.append("Runtime connector discovery router is not included in the API application.")

    try:
        package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"Could not parse package.json: {type(exc).__name__}")
        package = {}
    scripts = package.get("scripts", {})
    for name in ("lint", "build", "e2e"):
        if name not in scripts:
            failures.append(f"package.json is missing the '{name}' script.")

    for relative in (
        "e2e/helpers/auth.ts", "e2e/modules.spec.ts", "e2e/workflows.spec.ts",
        "e2e/evaluation.spec.ts", "e2e/api-playground.spec.ts",
        "e2e/runtime-connectors.spec.ts", "e2e/agent-profiles.spec.ts",
    ):
        target = root / relative
        if not target.is_file():
            failures.append(f"Browser E2E must bootstrap the local control-center session: {relative}")
        elif relative != "e2e/helpers/auth.ts" and "ensureAuthenticated" not in target.read_text(encoding="utf-8"):
            failures.append(f"Browser E2E must bootstrap the local control-center session: {relative}")

    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "project_version": project_version,
        "runtime_version": runtime_version,
        "roadmap_deliverables": len(ROADMAP_PATTERN.findall(roadmap)),
        "failures": failures,
        "warnings": warnings,
        "scope_note": "Static acceptance checks only; they do not establish runtime benchmark or training execution support.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="AXIOM repository root")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()
    report = inspect(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"AXIOM release smoke: {report['status'].upper()}")
        if "project_version" in report:
            print(f"Version: {report['project_version']} (runtime {report['runtime_version']})")
            print(f"Roadmap deliverables detected: {report['roadmap_deliverables']}")
        for issue in report.get("failures", []):
            print(f"FAIL: {issue}")
        for warning in report.get("warnings", []):
            print(f"WARN: {warning}")
        if report.get("scope_note"):
            print(report["scope_note"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
