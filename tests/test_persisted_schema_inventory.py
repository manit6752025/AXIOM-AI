from __future__ import annotations

import re
from pathlib import Path

from scripts import release_smoke


ROOT = Path(__file__).resolve().parents[1]


def _sources() -> dict[str, str]:
    paths = {
        "workflows": "axiom/core/workflows.py",
        "llm_profiles": "axiom/core/llm_profiles.py",
        "security_events": "axiom/core/security_events.py",
        "hardware_snapshots": "axiom/core/hardware_snapshots.py",
        "workspaces": "axiom/core/workspaces.py",
        "prediction_jobs": "axiom/core/prediction_jobs.py",
        "integrations": "axiom/core/integrations.py",
        "permissions": "axiom/core/tool_permissions.py",
        "registry": "axiom/models/registry.py",
        "workspace_metadata": "axiom/core/workspace_metadata.py",
        "fine_tuning": "axiom/core/fine_tuning.py",
        "model_studio": "axiom/core/model_studio.py",
        "runtime_traces": "axiom/core/runtime_traces.py",
    }
    return {name: (ROOT / path).read_text(encoding="utf-8") for name, path in paths.items()}


def test_storage_schema_inventory_matches_all_persisted_store_contracts() -> None:
    docs = (ROOT / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    source = _sources()

    assert "WORKFLOW_SCHEMA_VERSION = 2" in source["workflows"]
    assert "schema_version: 2" in docs
    assert "Schema-v1 records are accepted" in docs
    assert "SCHEMA_VERSION = 2" in source["llm_profiles"]
    assert ".axiom/llm-provider-profiles.json" in docs
    assert "LEGACY_SCHEMA_VERSION = 1" in source["security_events"]
    assert "SCHEMA_VERSION = 2" in source["security_events"]
    assert ".axiom/security-events.json" in docs
    assert "SCHEMA_VERSION = 1" in source["hardware_snapshots"]
    assert ".axiom/hardware-snapshots.json" in docs
    assert "_SCHEMA_VERSION = 1" in source["workspaces"]
    assert "~/.axiom/workspaces.json" in docs
    assert '"schema_version": 1' in source["prediction_jobs"]
    assert ".axiom/prediction_jobs.json" in docs

    for store in ("experiments.json", "evaluations.json", "agent-profiles.json", "models.json",
                  "integrations.json", "tool_permissions.json"):
        assert store in docs

    assert 'trust_status = data.get("trust_status", "unverified")' in source["integrations"]
    assert 'return _default_config()' in source["permissions"]
    assert 'legacy_resource_id(' in source["registry"]
    assert "_COLLECTION_FILES" in source["workspace_metadata"]
    assert "manifest.json" in docs and "ARCHIVE_SCHEMA_VERSION = 1" in source["workspace_metadata"]

    assert '"schema_version": 1' in source["fine_tuning"]
    assert '"schema_version": 1' in source["model_studio"]
    assert "does not yet enforce a strict future-version gate" in docs
    assert "Runtime traces are in-memory only" in docs
    assert "browser-memory-only" in docs


def test_storage_schema_release_contract_checks_current_code_and_docs() -> None:
    docs = (ROOT / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    assert release_smoke.persisted_schema_inventory_issues(docs, _sources()) == []


def test_storage_schema_release_contract_detects_stale_or_missing_migration_notes() -> None:
    docs = (ROOT / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    sources = _sources()
    stale = docs.replace("schema_version: 2", "schema_version: 1", 1)
    issues = release_smoke.persisted_schema_inventory_issues(stale, sources)
    assert any("workflows" in issue.lower() or "schema_version: 2" in issue for issue in issues)

    missing = docs.replace(".axiom/tool_permissions.json", ".axiom/removed.json")
    issues = release_smoke.persisted_schema_inventory_issues(missing, sources)
    assert any("tool_permissions.json" in issue for issue in issues)
