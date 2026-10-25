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


def test_model_registry_filters_are_wired_across_api_client_gui_and_e2e() -> None:
    api_source = (ROOT / "axiom" / "api" / "workspace.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "model-registry-filters.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.model_registry_filter_issues(api_source, client_source, app_source, e2e_source) == []


def test_model_registry_smoke_detects_disconnected_filter_controls() -> None:
    api_source = (ROOT / "axiom" / "api" / "workspace.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "model-registry-filters.spec.ts").read_text(encoding="utf-8")
    broken_app = app_source.replace('aria-label="Exact model tag"', 'aria-label="Removed filter"', 1)
    issues = release_smoke.model_registry_filter_issues(api_source, client_source, broken_app, e2e_source)
    assert any("Exact model tag" in issue for issue in issues)


def test_model_compatibility_report_is_wired_into_the_existing_models_page() -> None:
    core_source = (ROOT / "axiom" / "models" / "compatibility.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "workspace.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "model-registry-filters.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.model_compatibility_issues(core_source, api_source, client_source, app_source, e2e_source) == []


def test_model_compatibility_smoke_detects_missing_gui_action() -> None:
    core_source = (ROOT / "axiom" / "models" / "compatibility.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "workspace.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "model-registry-filters.spec.ts").read_text(encoding="utf-8")
    broken_app = app_source.replace("Check hardware fit for", "Removed fit action", 1)
    issues = release_smoke.model_compatibility_issues(core_source, api_source, client_source, broken_app, e2e_source)
    assert any("compatibility action" in issue for issue in issues)


def test_inference_benchmark_is_wired_from_runtime_gui_to_loopback_engine() -> None:
    core_source = (ROOT / "axiom" / "core" / "inference.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "inference.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalInferencePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "inference-benchmark.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.inference_benchmark_issues(core_source, api_source, client_source, panel_source, e2e_source) == []


def test_inference_benchmark_smoke_detects_unbounded_or_disconnected_gui() -> None:
    core_source = (ROOT / "axiom" / "core" / "inference.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "inference.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalInferencePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "inference-benchmark.spec.ts").read_text(encoding="utf-8")
    broken_panel = panel_source.replace("Benchmark this prompt", "Disconnected button", 1)
    issues = release_smoke.inference_benchmark_issues(core_source, api_source, client_source, broken_panel, e2e_source)
    assert any("Benchmark this prompt" in issue for issue in issues)


def test_inference_benchmark_checks_core_api_client_gui_and_browser_tests() -> None:
    core_source = (ROOT / "axiom" / "core" / "inference.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "inference.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalInferencePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "inference-benchmark.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.inference_benchmark_issues(core_source, api_source, client_source, panel_source, e2e_source) == []


def test_inference_benchmark_smoke_detects_disconnected_button() -> None:
    core_source = (ROOT / "axiom" / "core" / "inference.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "inference.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalInferencePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "inference-benchmark.spec.ts").read_text(encoding="utf-8")
    broken_panel = panel_source.replace("Benchmark this prompt", "Disconnected button", 1)
    issues = release_smoke.inference_benchmark_issues(core_source, api_source, client_source, broken_panel, e2e_source)
    assert any("Benchmark this prompt" in issue for issue in issues)


def test_runtime_trace_history_is_wired_across_api_client_gui_and_browser_tests() -> None:
    trace_core = (ROOT / "axiom" / "core" / "runtime_traces.py").read_text(encoding="utf-8")
    inference_api = (ROOT / "axiom" / "api" / "inference.py").read_text(encoding="utf-8")
    diagnostics_api = (ROOT / "axiom" / "api" / "diagnostics.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "RuntimeTracePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "runtime-traces.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.runtime_trace_issues(trace_core, inference_api, diagnostics_api, client_source, app_source, panel_source, e2e_source) == []


def test_runtime_trace_smoke_detects_missing_clear_route() -> None:
    trace_core = (ROOT / "axiom" / "core" / "runtime_traces.py").read_text(encoding="utf-8")
    inference_api = (ROOT / "axiom" / "api" / "inference.py").read_text(encoding="utf-8")
    diagnostics_api = (ROOT / "axiom" / "api" / "diagnostics.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "RuntimeTracePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "runtime-traces.spec.ts").read_text(encoding="utf-8")
    broken_api = diagnostics_api.replace('@router.delete("/runtime-traces")', '@router.post("/runtime-traces-clear")', 1)
    issues = release_smoke.runtime_trace_issues(trace_core, inference_api, broken_api, client_source, app_source, panel_source, e2e_source)
    assert any("Runtime trace history API is missing" in issue for issue in issues)


def test_engine_trusted_roots_and_native_execution_are_integrated_end_to_end() -> None:
    engine_source = (ROOT / "axiom" / "core" / "engine.py").read_text(encoding="utf-8")
    native_source = (ROOT / "axiom" / "core" / "native_engine.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "axiom-engine.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.engine_security_and_lifecycle_issues(engine_source, native_source, api_source, client_source, panel_source, e2e_source) == []


def test_engine_smoke_detects_path_and_native_execution_regressions() -> None:
    engine_source = (ROOT / "axiom" / "core" / "engine.py").read_text(encoding="utf-8")
    native_source = (ROOT / "axiom" / "core" / "native_engine.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "axiom-engine.spec.ts").read_text(encoding="utf-8")
    broken_engine = engine_source.replace("trusted-root escapes are not allowed", "unrestricted symlink target", 1)
    issues = release_smoke.engine_security_and_lifecycle_issues(broken_engine, native_source, api_source, client_source, panel_source, e2e_source)
    assert any("trusted-root escapes are not allowed" in issue for issue in issues)


def test_native_engine_smoke_rejects_missing_kv_cache_mechanics() -> None:
    engine_source = (ROOT / "axiom" / "core" / "engine.py").read_text(encoding="utf-8")
    native_source = (ROOT / "axiom" / "core" / "native_engine.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "axiom-engine.spec.ts").read_text(encoding="utf-8")
    broken_native = native_source.replace("past_key_values", "discarded_cache")
    issues = release_smoke.engine_security_and_lifecycle_issues(engine_source, broken_native, api_source, client_source, panel_source, e2e_source)
    assert any("past_key_values" in issue for issue in issues)


def test_native_stream_smoke_requires_cooperative_cancel_wiring() -> None:
    engine_source = (ROOT / "axiom" / "core" / "engine.py").read_text(encoding="utf-8")
    native_source = (ROOT / "axiom" / "core" / "native_engine.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "axiom-engine.spec.ts").read_text(encoding="utf-8")
    broken_engine = engine_source.replace("cancel_event=cancel_event", "cancel_event=None")
    issues = release_smoke.engine_security_and_lifecycle_issues(
        broken_engine, native_source, api_source, client_source, panel_source, e2e_source
    )
    assert any("cancel_event=cancel_event" in issue for issue in issues)


def test_adapter_preflight_is_wired_across_engine_api_client_gui_and_e2e() -> None:
    engine_source = (ROOT / "axiom" / "core" / "engine.py").read_text(encoding="utf-8")
    native_source = (ROOT / "axiom" / "core" / "native_engine.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "axiom-engine.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.engine_security_and_lifecycle_issues(
        engine_source, native_source, api_source, client_source, panel_source, e2e_source
    ) == []


def test_native_stop_sequences_are_integrated_and_regression_gated() -> None:
    engine_source = (ROOT / "axiom" / "core" / "engine.py").read_text(encoding="utf-8")
    native_source = (ROOT / "axiom" / "core" / "native_engine.py").read_text(encoding="utf-8")
    api_source = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    panel_source = (ROOT / "src" / "components" / "LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    e2e_source = (ROOT / "e2e" / "axiom-engine.spec.ts").read_text(encoding="utf-8")

    assert release_smoke.engine_security_and_lifecycle_issues(
        engine_source, native_source, api_source, client_source, panel_source, e2e_source
    ) == []

    broken_panel = panel_source.replace("stop_sequences: stopSequences,", "stop_sequence: stopSequences,")
    issues = release_smoke.engine_security_and_lifecycle_issues(
        engine_source, native_source, api_source, client_source, broken_panel, e2e_source
    )
    assert any("stop_sequences: stopSequences" in issue for issue in issues)


def test_api_playground_smoke_checks_typed_get_only_contract() -> None:
    component = (
        "playgroundRead allowlist GET only Arbitrary URLs mutating methods"
    )
    client = (
        "PlaygroundEndpoint playgroundEndpointPaths playgroundRead method: \"GET\""
    )
    assert release_smoke.api_playground_issues(component, client) == []


def test_api_playground_smoke_rejects_unbounded_client_or_missing_disclosure() -> None:
    component = "playgroundRead allowlist"
    client = "playgroundRead fetch(any_url)"
    issues = release_smoke.api_playground_issues(component, client)
    assert any("typed endpoint allowlist" in issue for issue in issues)
    assert any("disclose the allowlist" in issue for issue in issues)
