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
        for i in range(1, 141)
    )
    assert release_smoke.roadmap_issues(markdown) == []


def test_roadmap_audit_accepts_implemented_status_with_code_scan_review_pending() -> None:
    markdown = "\n".join(
        f"- [R{i:03d} · {'Planned' if i % 2 else 'Implemented; tests passing'}] Requirement {i}."
        for i in range(1, 140)
    )
    markdown += "\n- [R140 · Implemented; tests passing; code-scan review pending] CodeQL review remains an owner action."
    assert release_smoke.roadmap_issues(markdown) == []


def test_roadmap_audit_detects_duplicates_missing_ids_and_unknown_status() -> None:
    markdown = "- [R001 · Maybe] A\n- [R001 · Planned] Duplicate\n"
    issues = release_smoke.roadmap_issues(markdown)
    assert any("Expected 140" in item for item in issues)
    assert any("Missing roadmap IDs" in item for item in issues)
    assert any("Duplicate roadmap IDs: R001" in item for item in issues)
    assert any("Unknown roadmap statuses: Maybe" in item for item in issues)


def test_resource_monitoring_contract_requires_opt_in_bounds_export_and_gui_history() -> None:
    cli = (ROOT / "axiom" / "cli" / "engine.py").read_text(encoding="utf-8")
    panel = (ROOT / "src" / "components" / "EngineTelemetryMonitor.tsx").read_text(encoding="utf-8")
    app = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e" / "telemetry-monitor.spec.ts").read_text(encoding="utf-8")
    core = (ROOT / "axiom" / "core" / "resource_telemetry.py").read_text(encoding="utf-8")
    api = (ROOT / "axiom" / "api" / "engine.py").read_text(encoding="utf-8")
    client = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    core_tests = (ROOT / "tests" / "test_resource_telemetry.py").read_text(encoding="utf-8")
    api_tests = (ROOT / "tests" / "test_resource_telemetry_api.py").read_text(encoding="utf-8")

    arguments = (cli, panel, app, e2e, core, api, client, core_tests, api_tests)
    assert release_smoke.resource_monitoring_issues(*arguments) == []
    issues = release_smoke.resource_monitoring_issues(
        cli, panel, app, e2e.replace('name: "Start monitoring"', 'name: "Begin monitoring"'),
        core, api, client, core_tests, api_tests,
    )
    assert any("Browser acceptance" in issue and "Start monitoring" in issue for issue in issues)


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
    broken_engine = engine_source.replace("cancel_event=cancel_event", "cancel_event=None").replace("cancel_event=cancel_event", "cancel_event=None")
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

    broken_panel = panel_source.replace("stop_sequences: stopSequences,", "stop_sequence: stopSequences,").replace("stop_sequences: stopSequences,", "stop_sequence: stopSequences,")
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



def test_generated_typed_api_examples_are_wired_from_openapi_to_cli_gui_and_docs() -> None:
    core = (ROOT / "axiom/core/api_examples.py").read_text(encoding="utf-8")
    api = (ROOT / "axiom/api/examples.py").read_text(encoding="utf-8")
    server = (ROOT / "axiom/api/server.py").read_text(encoding="utf-8")
    cli = (ROOT / "axiom/cli/api.py").read_text(encoding="utf-8")
    cli_main = (ROOT / "axiom/cli/main.py").read_text(encoding="utf-8")
    client = (ROOT / "src/api/client.ts").read_text(encoding="utf-8")
    component = (ROOT / "src/components/ApiPlayground.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/api-playground.spec.ts").read_text(encoding="utf-8")
    tests = (ROOT / "tests/test_api_examples.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    assert release_smoke.api_contract_examples_issues(
        core, api, server, cli, cli_main, client, component, e2e, tests, docs,
    ) == []


def test_generated_api_example_smoke_detects_missing_secret_redaction_or_gui_binding() -> None:
    core = (ROOT / "axiom/core/api_examples.py").read_text(encoding="utf-8")
    api = (ROOT / "axiom/api/examples.py").read_text(encoding="utf-8")
    server = (ROOT / "axiom/api/server.py").read_text(encoding="utf-8")
    cli = (ROOT / "axiom/cli/api.py").read_text(encoding="utf-8")
    cli_main = (ROOT / "axiom/cli/main.py").read_text(encoding="utf-8")
    client = (ROOT / "src/api/client.ts").read_text(encoding="utf-8")
    component = (ROOT / "src/components/ApiPlayground.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/api-playground.spec.ts").read_text(encoding="utf-8")
    tests = (ROOT / "tests/test_api_examples.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    broken_core = core.replace("x-axiom-redacted", "x-no-redaction")
    issues = release_smoke.api_contract_examples_issues(
        broken_core, api, server, cli, cli_main, client, component, e2e, tests, docs,
    )
    assert any("x-axiom-redacted" in issue for issue in issues)



def test_hardware_snapshot_release_smoke_requires_opt_in_api_cli_and_gui_contract() -> None:
    core = (ROOT / "axiom/core/hardware_snapshots.py").read_text(encoding="utf-8")
    api = (ROOT / "axiom/api/hardware_snapshots.py").read_text(encoding="utf-8")
    server = (ROOT / "axiom/api/server.py").read_text(encoding="utf-8")
    cli_main = (ROOT / "axiom/cli/main.py").read_text(encoding="utf-8")
    client = (ROOT / "src/api/client.ts").read_text(encoding="utf-8")
    panel = (ROOT / "src/components/HardwareSnapshotPanel.tsx").read_text(encoding="utf-8")
    app = (ROOT / "src/App.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/hardware-snapshots.spec.ts").read_text(encoding="utf-8")
    tests = "\\n".join((ROOT / path).read_text(encoding="utf-8") for path in (
        "tests/test_hardware_snapshots.py",
        "tests/test_hardware_snapshots_api.py",
        "tests/test_hardware_snapshots_cli.py",
    ))
    docs = "\\n".join((ROOT / path).read_text(encoding="utf-8") for path in (
        "docs/API.md", "docs/wiki/CLI-Reference.md", "docs/Storage-Schemas-and-Migrations.md",
    ))
    assert release_smoke.hardware_snapshot_issues(
        core, api, server, cli_main, client, panel, app, e2e, tests, docs,
    ) == []


def test_hardware_snapshot_release_smoke_detects_missing_opt_in_contract() -> None:
    core = (ROOT / "axiom/core/hardware_snapshots.py").read_text(encoding="utf-8")
    api = (ROOT / "axiom/api/hardware_snapshots.py").read_text(encoding="utf-8")
    server = (ROOT / "axiom/api/server.py").read_text(encoding="utf-8")
    cli_main = (ROOT / "axiom/cli/main.py").read_text(encoding="utf-8")
    client = (ROOT / "src/api/client.ts").read_text(encoding="utf-8")
    panel = (ROOT / "src/components/HardwareSnapshotPanel.tsx").read_text(encoding="utf-8")
    app = (ROOT / "src/App.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/hardware-snapshots.spec.ts").read_text(encoding="utf-8")
    tests = (ROOT / "tests/test_hardware_snapshots.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    broken_core = core.replace("raise HardwareSnapshotOptInRequired(", "raise MissingOptInGuard(")
    issues = release_smoke.hardware_snapshot_issues(
        broken_core, api, server, cli_main, client, panel, app, e2e, tests, docs,
    )
    assert any("HardwareSnapshotOptInRequired" in item for item in issues)


def test_benchmark_resource_trend_release_contract_is_wired_end_to_end() -> None:
    panel = (ROOT / "src/components/LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    chart = (ROOT / "src/components/BenchmarkResourceTrends.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/axiom-engine.spec.ts").read_text(encoding="utf-8")
    assert release_smoke.benchmark_resource_trend_issues(panel, chart, e2e) == []


def test_benchmark_resource_trend_release_contract_detects_unbounded_or_missing_charts() -> None:
    panel = (ROOT / "src/components/LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    chart = (ROOT / "src/components/BenchmarkResourceTrends.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/axiom-engine.spec.ts").read_text(encoding="utf-8")
    broken_chart = chart.replace("Not reported", "unknown metric")
    failures = release_smoke.benchmark_resource_trend_issues(panel, broken_chart, e2e)
    assert any("Not reported" in issue for issue in failures)


def test_telemetry_export_privacy_release_contract_requires_json_csv_and_clear() -> None:
    monitor = (ROOT / "src/components/EngineTelemetryMonitor.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/telemetry-monitor.spec.ts").read_text(encoding="utf-8")
    docs = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    assert release_smoke.telemetry_export_privacy_issues(monitor, e2e, docs) == []


def test_telemetry_export_privacy_contract_detects_pid_or_missing_csv_controls() -> None:
    monitor = (ROOT / "src/components/EngineTelemetryMonitor.tsx").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/telemetry-monitor.spec.ts").read_text(encoding="utf-8")
    docs = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    broken = monitor.replace('anchor.download = "axiom-resource-telemetry.csv"', "anchor.download = 'wrong.csv'")
    failures = release_smoke.telemetry_export_privacy_issues(broken, e2e, docs)
    assert any("axiom-resource-telemetry.csv" in issue for issue in failures)


def test_cross_platform_package_release_contract_requires_wheel_sdist_and_install() -> None:
    verifier = (ROOT / "scripts/verify_package_artifacts.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/quality-gates.yml").read_text(encoding="utf-8")
    tests = (ROOT / "tests/test_package_artifact_verification.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs/Packaging.md").read_text(encoding="utf-8")
    assert release_smoke.package_verification_issues(verifier, workflow, tests, docs) == []


def test_cross_platform_package_release_contract_detects_missing_sdist_build() -> None:
    verifier = (ROOT / "scripts/verify_package_artifacts.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/quality-gates.yml").read_text(encoding="utf-8")
    tests = (ROOT / "tests/test_package_artifact_verification.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs/Packaging.md").read_text(encoding="utf-8")
    broken_workflow = workflow.replace("python -m build --sdist --wheel --outdir dist", "python -m build --wheel --outdir dist")
    failures = release_smoke.package_verification_issues(verifier, broken_workflow, tests, docs)
    assert any("sdist" in issue for issue in failures)


def test_persisted_schema_inventory_release_contract_matches_current_stores() -> None:
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
        "resource_telemetry": "axiom/core/resource_telemetry.py",
    }
    sources = {name: (ROOT / path).read_text(encoding="utf-8") for name, path in paths.items()}
    docs = (ROOT / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    assert release_smoke.persisted_schema_inventory_issues(docs, sources) == []


def test_persisted_schema_inventory_detects_stale_versions_and_missing_store_notes() -> None:
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
        "resource_telemetry": "axiom/core/resource_telemetry.py",
    }
    sources = {name: (ROOT / path).read_text(encoding="utf-8") for name, path in paths.items()}
    docs = (ROOT / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    stale = docs.replace("schema_version: 2", "schema_version: 1", 1).replace(".axiom/tool_permissions.json", ".axiom/missing.json")
    failures = release_smoke.persisted_schema_inventory_issues(stale, sources)
    assert any("schema_version: 2" in failure for failure in failures)
    assert any("tool_permissions.json" in failure for failure in failures)


def test_supply_chain_contract_requires_spdx_and_explicit_dependency_audit() -> None:
    core = (
        'def generate_spdx_sbom(\n"SPDX-2.3"\n"PACKAGE-MANAGER"\n"DEPENDS_ON"\n'
        'MAX_DISTRIBUTIONS = 4096\nhost_paths_collected\nnetwork_access_performed\n'
        'def audit_dependency_vulnerabilities(\nallow_network: bool\nexplicit network authorization\n'
        'MAX_AUDIT_OUTPUT_BYTES\n"--local"\n"--format", "json"\nsubprocess.TimeoutExpired\nPIP_INDEX_URL\n"unavailable"'
    )
    api = '@router.get("/sbom")\n@router.post("/dependency-audit")\nallow_network=request.allow_network\ntimeout_seconds=request.timeout_seconds'
    cli = '@security_app.command("sbom")\n@security_app.command("vulnerabilities")\n"--allow-network"\nwrite_report_new('
    client = "SoftwareBillOfMaterials\nDependencyVulnerabilityReport\nsecuritySoftwareBillOfMaterials:\nscanDependencyVulnerabilities:"
    gui = "exportSoftwareBillOfMaterials\nscanDependencyVulnerabilities\nwindow.confirm(\nExport SPDX 2.3 SBOM\nScan dependencies (network)\nunavailable scanner is never described as a clean scan"
    core_tests = (
        "test_spdx_sbom_contains_package_urls_versions_and_installed_dependency_relationships\n"
        "test_spdx_sbom_rejects_unbounded_or_invalid_installed_metadata\n"
        "test_report_export_rejects_symlinked_parent_and_existing_symlink\n"
        "test_vulnerability_scan_requires_explicit_network_authorization\n"
        "test_vulnerability_scan_reports_unavailable_scanner_without_claiming_clean\n"
        "test_vulnerability_scan_invokes_scanner_only_after_authorization_and_drops_secret_environment\n"
        "test_vulnerability_scan_does_not_report_clean_on_timeout"
    )
    api_tests = (
        "test_security_supply_chain_api_returns_spdx_sbom\n"
        "test_security_supply_chain_api_requires_explicit_network_authorization\n"
        "test_security_supply_chain_api_runs_only_authorized_bounded_dependency_scan"
    )
    e2e = (
        'Export SPDX 2.3 SBOM\nScan dependencies (network)\npage.waitForEvent("dialog")\n'
        'axiom-installed-environment.spdx.json\naxiom-dependency-vulnerability-report.json\nallow_network: true'
    )
    docs = "SPDX 2.3\npip-audit\nnetwork authorization\nunavailable\ninstalled Python"

    assert release_smoke.supply_chain_issues(core, api, cli, client, gui, core_tests, api_tests, e2e, docs) == []
    failures = release_smoke.supply_chain_issues(
        core.replace("explicit network authorization", ""),
        api, cli, client, gui, core_tests, api_tests, e2e, docs,
    )
    assert any("network authorization" in failure for failure in failures)


def test_supply_chain_contract_requires_spdx_and_explicit_dependency_audit() -> None:
    core = (
        'def generate_spdx_sbom(\n"SPDX-2.3"\n"PACKAGE-MANAGER"\n"DEPENDS_ON"\n'
        'MAX_DISTRIBUTIONS = 4096\nhost_paths_collected\nnetwork_access_performed\n'
        'def audit_dependency_vulnerabilities(\nallow_network: bool\nexplicit network authorization\n'
        'MAX_AUDIT_OUTPUT_BYTES\n"--local"\n"--format", "json"\nsubprocess.TimeoutExpired\nPIP_INDEX_URL\n"unavailable"'
    )
    api = '@router.get("/sbom")\n@router.post("/dependency-audit")\nallow_network=request.allow_network\ntimeout_seconds=request.timeout_seconds'
    cli = '@security_app.command("sbom")\n@security_app.command("vulnerabilities")\n"--allow-network"\nwrite_report_new('
    client = "SoftwareBillOfMaterials\nDependencyVulnerabilityReport\nsecuritySoftwareBillOfMaterials:\nscanDependencyVulnerabilities:"
    gui = "exportSoftwareBillOfMaterials\nscanDependencyVulnerabilities\nwindow.confirm(\nExport SPDX 2.3 SBOM\nScan dependencies (network)\nunavailable scanner is never described as a clean scan"
    core_tests = (
        "test_spdx_sbom_contains_package_urls_versions_and_installed_dependency_relationships\n"
        "test_spdx_sbom_rejects_unbounded_or_invalid_installed_metadata\n"
        "test_report_export_rejects_symlinked_parent_and_existing_symlink\n"
        "test_vulnerability_scan_requires_explicit_network_authorization\n"
        "test_vulnerability_scan_reports_unavailable_scanner_without_claiming_clean\n"
        "test_vulnerability_scan_invokes_scanner_only_after_authorization_and_drops_secret_environment\n"
        "test_vulnerability_scan_does_not_report_clean_on_timeout"
    )
    api_tests = (
        "test_security_supply_chain_api_returns_spdx_sbom\n"
        "test_security_supply_chain_api_requires_explicit_network_authorization\n"
        "test_security_supply_chain_api_runs_only_authorized_bounded_dependency_scan"
    )
    e2e = (
        'Export SPDX 2.3 SBOM\nScan dependencies (network)\npage.waitForEvent("dialog")\n'
        'axiom-installed-environment.spdx.json\naxiom-dependency-vulnerability-report.json\nallow_network: true'
    )
    docs = "SPDX 2.3\npip-audit\nnetwork authorization\nunavailable\ninstalled Python\nsecurity-audit\npip-audit>=2.10.1,<3"

    assert release_smoke.supply_chain_issues(core, api, cli, client, gui, core_tests, api_tests, e2e, docs) == []
    failures = release_smoke.supply_chain_issues(
        core.replace("explicit network authorization", ""),
        api, cli, client, gui, core_tests, api_tests, e2e, docs,
    )
    assert any("network authorization" in failure for failure in failures)


def test_dataset_snapshot_release_contract_requires_immutable_store_and_lineage_coverage() -> None:
    from scripts.release_smoke import dataset_snapshot_issues

    core = (
        "class DatasetSnapshotStore\nMAX_SNAPSHOTS = 250\n"
        "MAX_DATASET_BYTES = 1024 * 1024 * 1024\nMAX_TOTAL_SNAPSHOT_BYTES = 10 * 1024 * 1024 * 1024\n"
        "def create_snapshot(\ndef verify(\nfile_lock(\natomic_write_text(\n"
        "record_sha256\nparent_manifest_sha256\nobject_name"
    )
    api = (
        "class WorkspaceDatasetSnapshotInput(BaseModel)\n"
        '@router.get("/dataset-snapshots")\n@router.post("/dataset-snapshots", status_code=201)\n'
        '@router.get("/dataset-snapshots/{snapshot_id}/verify")\n'
        "DatasetSnapshotStore(active_workspace_root())\nopen_regular_file(candidate)"
    )
    cli = (
        '@dataset_app.command("snapshot")\n@dataset_app.command("snapshots")\n'
        '@dataset_app.command("snapshot-verify")\nDatasetSnapshotStore(workspace)\n'
        "find_existing_relative_path(data_root"
    )
    client = (
        "DatasetSnapshotManifest\nDatasetSnapshotHistory\nDatasetSnapshotVerification\n"
        "datasetSnapshots:\ncreateDatasetSnapshot:\nverifyDatasetSnapshot:"
    )
    gui = (
        "snapshotHistory\nsnapshotParentId\nrunSnapshot\nverifySnapshot\n"
        "createDatasetSnapshot(\nverifyDatasetSnapshot(\nParent dataset snapshot\n"
        "Immutable snapshot history\nSnapshot"
    )
    core_tests = (
        "test_snapshot_stores_content_addressed_read_only_object_and_value_free_manifest\n"
        "test_identical_content_is_deduplicated_and_parent_lineage_is_verifiable\n"
        "test_parent_must_be_a_valid_earlier_snapshot_in_the_same_workspace\n"
        "test_snapshot_enforces_source_type_format_and_size_bounds\n"
        "test_snapshot_rejects_tampered_manifest_index_and_hash_chain\n"
        "test_snapshot_verification_detects_modified_object_bytes\n"
        "test_snapshot_rejects_symlinked_workspace_metadata"
    )
    api_tests = (
        "test_dataset_snapshot_api_creates_lists_and_verifies_workspace_snapshots\n"
        "test_dataset_snapshot_api_rejects_unsafe_sources_parent_and_byte_overflow\n"
        "test_dataset_snapshot_api_links_parent_manifest_to_prior_snapshot\n"
        "test_dataset_snapshot_api_returns_not_found_for_unknown_snapshot"
    )
    cli_tests = (
        "test_dataset_snapshot_cli_creates_lists_and_verifies_immutable_history\n"
        "test_dataset_snapshot_cli_rejects_files_outside_workspace_data"
    )
    e2e = (
        "Datasets creates immutable content-addressed snapshots\nParent dataset snapshot\n"
        "Verify\nparent_snapshot_id: firstId\nSnapshot content hash and lineage verified."
    )
    docs = "content-addressed\nhash-chained\nread-only\nparent_snapshot_id\nnot a signed\naxiom dataset snapshot"

    args = (core, api, cli, client, gui, core_tests, api_tests, cli_tests, e2e, docs)
    assert dataset_snapshot_issues(*args) == []
    failures = dataset_snapshot_issues(
        core.replace("file_lock(", ""),
        api, cli, client, gui, core_tests, api_tests, cli_tests, e2e, docs,
    )
    assert any("file_lock(" in failure for failure in failures)



def test_release_smoke_initializes_shared_api_client_before_snapshot_contract(monkeypatch, tmp_path) -> None:
    import ast

    source = (ROOT / "scripts/release_smoke.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    inspect_fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "inspect")
    assignments = {}
    for node in inspect_fn.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "client":
                    assignments[target.id] = node.lineno
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "snapshot_client":
                    snapshot_line = node.lineno
    assert assignments["client"] < snapshot_line


def test_benchmark_profile_release_contract_checks_all_integration_layers() -> None:
    from scripts.release_smoke import benchmark_profile_issues

    core = (ROOT / "axiom/core/benchmark_profiles.py").read_text(encoding="utf-8")
    api = (ROOT / "axiom/api/engine.py").read_text(encoding="utf-8")
    cli = (ROOT / "axiom/cli/engine.py").read_text(encoding="utf-8")
    client = (ROOT / "src/api/client.ts").read_text(encoding="utf-8")
    panel = (ROOT / "src/components/LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    core_tests = (ROOT / "tests/test_benchmark_profiles.py").read_text(encoding="utf-8")
    api_tests = (ROOT / "tests/test_benchmark_profiles_api.py").read_text(encoding="utf-8")
    cli_tests = (ROOT / "tests/test_axiom_engine_cli.py").read_text(encoding="utf-8")
    e2e = (ROOT / "e2e/axiom-engine.spec.ts").read_text(encoding="utf-8")
    docs = (ROOT / "docs/Inference-Benchmarking.md").read_text(encoding="utf-8")

    assert benchmark_profile_issues(core, api, cli, client, panel, core_tests, api_tests, cli_tests, e2e, docs) == []
    assert any("prompt_sha256" in issue for issue in benchmark_profile_issues(
        core.replace("prompt_sha256", "prompt_hash"), api, cli, client, panel,
        core_tests, api_tests, cli_tests, e2e, docs,
    ))
