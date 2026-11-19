from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("axiom_release_smoke", ROOT / "scripts" / "release_smoke.py")
assert SPEC and SPEC.loader
release_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_smoke)


def test_roadmap_audit_accepts_all_unique_deliverables() -> None:
    markdown = "\n".join(
        f"- [R{i:03d} · {'Planned' if i % 2 else 'Implemented; tests passing'}] Requirement {i}."
        for i in range(1, 121)
    )
    assert release_smoke.roadmap_issues(markdown) == []


def test_roadmap_audit_detects_duplicates_missing_ids_and_unknown_status() -> None:
    markdown = "- [R001 · Maybe] A\n- [R001 · Planned] Duplicate\n"
    issues = release_smoke.roadmap_issues(markdown)
    assert any("Expected 120" in item for item in issues)
    assert any("Missing roadmap IDs" in item for item in issues)
    assert any("Duplicate roadmap IDs: R001" in item for item in issues)
    assert any("Unknown roadmap statuses: Maybe" in item for item in issues)


def test_gui_page_registration_requires_union_nav_metadata_and_switch() -> None:
    complete = '''
  | "api-playground"
  { id: "api-playground", label: "API Playground" }
  "api-playground": { label: "API Playground" }
  case "api-playground": return <ApiPlayground />;
'''
    assert release_smoke.page_registration_issues(complete, "api-playground") == []
    incomplete = '''
  | "api-playground"
  { id: "api-playground" }
'''
    issues = release_smoke.page_registration_issues(incomplete, "api-playground")
    assert any("page metadata" in item for item in issues)
    assert any("render switch" in item for item in issues)


def test_agent_profile_page_registration_requires_all_four_gui_links() -> None:
    source = '''
  | "agent-profiles"
    id: "agent-profiles"
  "agent-profiles": { label: "Agent Profiles" }
    case "agent-profiles": return <AgentProfiles />;
'''
    assert release_smoke.page_registration_issues(source, "agent-profiles") == []


def test_module_explorer_requires_new_gui_pages_in_the_catalog() -> None:
    complete = 'function ModulesPage() { const entries = [{ id: "api-playground" }, { id: "agent-profiles" }]; }'
    assert release_smoke.module_catalog_issues(complete) == []
    incomplete = "function ModulesPage() { const entries = [{ id: \"api-playground\" }]; }"
    assert any("agent-profiles" in issue for issue in release_smoke.module_catalog_issues(incomplete))
