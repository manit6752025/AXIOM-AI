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
    "Implemented; tests passing; code-scan review pending",
}


def roadmap_issues(markdown: str) -> list[str]:
    entries = ROADMAP_PATTERN.findall(markdown)
    ids = [item[0] for item in entries]
    expected = {f"R{i:03d}" for i in range(1, 141)}
    issues: list[str] = []
    missing = sorted(expected - set(ids))
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    unexpected = sorted(set(ids) - expected)
    unknown_statuses = sorted({status for _, status in entries if status not in VALID_STATUSES})
    if len(entries) != 140:
        issues.append(f"Expected 140 roadmap entries; found {len(entries)}.")
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


def model_registry_filter_issues(api_source: str, client_source: str, app_source: str, e2e_source: str) -> list[str]:
    """Check the model registry filter path from its API through GUI and browser tests."""
    issues: list[str] = []
    for token in (
        'search: str = Query(default="", max_length=160)',
        "favorite_only: bool = False",
        "format: str | None = Query",
        "tag: str | None = Query",
        'sort_by: Literal["name", "source", "format", "parameters", "quantization", "favorite"]',
        'sort_order: Literal["asc", "desc"]',
        '"has_more": offset + limit < total',
    ):
        if token not in api_source:
            issues.append("Model registry API is missing a supported filter/sort/pagination contract: " + token)
    for token in (
        'query.set("search"',
        'query.set("favorite_only"',
        'query.set("format"',
        'query.set("tag"',
        'query.set("sort_by"',
        'query.set("sort_order"',
    ):
        if token not in client_source:
            issues.append("Typed API client is missing model registry query serialization: " + token)
    for token in (
        'aria-label="Search model registry"',
        'aria-label="Exact model format"',
        'aria-label="Exact model tag"',
        'aria-label="Favorites only"',
        "SAVED_MODEL_FILTERS_KEY",
        "localStorage.setItem(SAVED_MODEL_FILTERS_KEY",
        "setAppliedFilters] = useState<ModelRegistryFilters>",
        "modelsPage(models.length, 25, appliedFilters)",
    ):
        if token not in app_source:
            issues.append("Models GUI is missing integrated filter/preset behavior: " + token)
    for token in (
        "model registry filters and saved presets",
        'searchParams.get("favorite_only")',
        'searchParams.get("sort_by")',
        "Save current filters",
        'getByRole("button", { name: "Apply", exact: true })',
    ):
        if token not in e2e_source:
            issues.append("Model registry browser acceptance coverage is missing: " + token)
    return issues


def model_compatibility_issues(core_source: str, api_source: str, client_source: str, app_source: str, e2e_source: str) -> list[str]:
    """Check model-fit estimates are connected from the backend into the existing Models page."""
    issues: list[str] = []
    for token in (
        "def estimate_model_compatibility(",
        '"estimated_weight_gb": estimated_weights',
        '"estimated_runtime_memory_gb": estimated_runtime',
        '"system_ram_fit": ram_fit',
        '"gpu_memory_fit": vram_fit',
        '"format_compatibility": format_compatibility',
    ):
        if token not in core_source:
            issues.append("Model fit engine is missing an evidence-labelled estimate: " + token)
    for token in (
        '@router.get("/models/{model_id}/compatibility")',
        "estimate_model_compatibility(model, detect_hardware())",
        'return {"report": report}',
    ):
        if token not in api_source:
            issues.append("Model fit API is missing a backend connection: " + token)
    for token in (
        "export type WorkspaceModelCompatibility",
        "modelCompatibility: (id: string)",
        "/compatibility",
    ):
        if token not in client_source:
            issues.append("Typed API client is missing the model fit endpoint: " + token)
    for token in (
        'aria-label={`Check hardware fit for ${model.name}`}',
    ):
        if token not in app_source:
            issues.append("Models GUI is missing the compatibility action: " + token)
    for token in (
        'getByRole("button", { name: "Check hardware fit for Qwen Seven" })',
        'getByRole("heading", { name: "Hardware fit · Qwen Seven" })',
        'getByText("UNLIKELY", { exact: true })',
    ):
        if token not in e2e_source:
            issues.append("Model fit browser coverage is missing: " + token)
    return issues


def benchmark_profile_issues(profile_core: str, api_source: str, cli_source: str, client_source: str, panel_source: str, core_tests: str, api_tests: str, cli_tests: str, e2e_source: str, docs_source: str) -> list[str]:
    """Require saved benchmark provenance to be connected to tests and UX."""
    checks = (
        ("Profile store", profile_core, ("class BenchmarkProfileStore", "def create(", "def list(", "def compare(", "prompt_sha256", "model_config_sha256", "_hardware_record(", "_software_record(", "MAX_PROFILES = 100", "MAX_STORE_BYTES", "file_lock(", "atomic_write_text(", "stat.S_ISREG", "adapter_used")),
        ("Benchmark API", api_source, ('profile_name: str | None', 'BenchmarkProfileStore().create(', '@router.get("/benchmark-profiles")', '@router.get("/benchmark-profiles/compare")')),
        ("Benchmark CLI", cli_source, ('"--profile-name"', '@engine_app.command("benchmark-profiles")', '@engine_app.command("benchmark-compare")', 'BenchmarkProfileStore().create(')),
        ("Typed API client", client_source, ("AxiomEngineBenchmarkProfile", "AxiomEngineBenchmarkProfileComparison", "axiomEngineBenchmarkProfiles:", "compareAxiomEngineBenchmarkProfiles:", "profile_name?: string")),
        ("Runtime GUI", panel_source, ('aria-label="Benchmark profile name"', "profile_name: benchmarkProfileName.trim()", "Saved hardware-specific profiles", "Compare selected profiles", "Descriptive comparison only", "Prompt content and model paths are never saved")),
        ("Core regression tests", core_tests, ("test_profile_store_records_reproducible_provenance_without_prompt_or_model_paths", "test_profiles_compare_metrics_only_when_hardware_software_model_and_workload_match", "test_profile_store_rejects_duplicate_names_and_invalid_profile_inputs", "test_profile_store_rejects_tampering_unknown_fields_and_future_schema", "test_profile_store_rejects_symlinked_store_paths", "test_profile_comparison_requires_two_distinct_existing_ids")),
        ("API regression tests", api_tests, ("test_native_benchmark_endpoint_saves_profile_only_when_requested", "test_native_benchmark_endpoint_does_not_create_profile_by_default", "test_native_benchmark_profile_routes_list_and_compare_workspace_records", "test_native_benchmark_profile_name_is_validated_before_execution")),
        ("CLI regression tests", cli_tests, ("test_native_benchmark_profile_list_cli_uses_workspace_store", "test_native_benchmark_profile_compare_cli_reports_comparability", "test_native_benchmark_cli_saves_named_profile_when_requested")),
        ("Browser acceptance", e2e_source, ("Benchmark profile name", "Qwen CPU baseline", "profile_name", "/api/engine/benchmark-profiles", "/api/engine/benchmark-profiles/compare", "Saved reproducibility profile", "Controlled comparison")),
        ("Benchmark documentation", docs_source, ("axiom engine benchmark --profile-name", "axiom engine benchmark-profiles", "axiom engine benchmark-compare", "prompt fingerprint", "hardware and software", "model paths are not persisted")),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Hardware-specific benchmark profile {label} is missing: " + token)
    return failures

def inference_benchmark_issues(core_source: str, api_source: str, client_source: str, panel_source: str, e2e_source: str) -> list[str]:
    """Check benchmark implementation connects the measured loopback runtime to Runtime UI."""
    issues: list[str] = []
    for token in (
        "def run_local_inference_benchmark(",
        "ThreadPoolExecutor",
        "measured_requests > 10",
        '"completion_tokens_per_second":',
        '"prompt_returned": False',
        '"responses_returned": False',
        '"results_persisted": False',
    ):
        if token not in core_source:
            issues.append("Local inference benchmark engine is missing a bounded measurement contract: " + token)
    for token in (
        '@router.post("/benchmark")',
        "run_local_inference_benchmark(",
        "request.iterations * request.concurrency > 10",
    ):
        if token not in api_source:
            issues.append("Local inference benchmark API is missing: " + token)
    for token in (
        "export type LocalInferenceBenchmark",
        '"/api/inference/benchmark"',
        "runLocalInferenceBenchmark:",
    ):
        if token not in client_source:
            issues.append("Typed inference benchmark client is missing: " + token)
    for token in (
        'aria-labelledby="inference-benchmark-title"',
        'aria-label="Benchmark iterations"',
        'aria-label="Benchmark concurrency"',
        'onClick={() => void benchmarkLocalModel()}',
        "runLocalInferenceBenchmark({",
        "Benchmark this prompt",
        "Inference benchmark result",
    ):
        if token not in panel_source:
            issues.append("Runtime GUI benchmark controls/results are missing: " + token)
    for token in (
        "Measured local inference benchmark",
        '"/api/inference/benchmark"',
        "blocks a requested workload above the ten measured-request limit",
        "42.857 tokens/s",
    ):
        if token not in e2e_source:
            issues.append("Inference benchmark browser acceptance coverage is missing: " + token)
    return issues


def structured_inference_issues(
    schema_core: str, inference_core: str, inference_api: str, client_source: str,
    cli_source: str, panel_source: str, schema_tests: str, core_tests: str,
    api_tests: str, cli_tests: str, e2e_source: str, docs_source: str,
) -> list[str]:
    """Check JSON Schema constrained local inference across runtime, API, CLI and GUI."""
    issues: list[str] = []
    checks = (
        ("schema validator", schema_core, (
            "MAX_SCHEMA_BYTES = 32 * 1024", "MAX_SCHEMA_DEPTH = 8",
            "MAX_SCHEMA_NODES = 128", "MAX_OUTPUT_CHARS = 65_536",
            "def validate_json_schema(", "def parse_and_validate_json_output(",
            '_PROPERTY_NAME = re.compile(r"^[^\\x00]{1,128}$")',
            "additionalProperties", "object_pairs_hook=unique_pairs",
        )),
        ("runtime", inference_core, (
            "response_format: dict[str, Any] | None = None",
            "normalized_schema = validate_json_schema(response_format)",
            'normalized_schema.pop("$schema", None)',
            'payload["format"] = normalized_schema',
            "Return only one JSON value",
        )),
        ("API", inference_api, (
            'class StructuredGenerateRequest(GenerateRequest):',
            '@router.post("/generate/structured")',
            "structured_output_token_limit", "parse_and_validate_json_output(",
            '"structured_output_validated": True',
        )),
        ("typed client", client_source, (
            "export type LocalInferenceStructuredInput",
            "generateStructuredLocalInference:",
            '"/api/inference/generate/structured"',
        )),
        ("CLI", cli_source, (
            '"--schema-file"', "def _load_json_schema_file(", "O_NOFOLLOW",
            "parse_and_validate_json_output(", "structured_output_validated",
        )),
        ("Runtime GUI", panel_source, (
            "Structured JSON output · optional", "JSON Schema for structured output",
            "generateStructuredLocalInference(", "SCHEMA VALIDATED",
        )),
        ("schema tests", schema_tests, (
            "test_schema_subset_normalizes_to_supported_draft",
            "test_schema_rejects_unsupported_or_unbounded_keywords",
            "test_output_parser_rejects_duplicate_non_finite_invalid_or_wrong_type_values",
        )),
        ("core tests", core_tests, (
            "test_structured_generation_passes_validated_schema_to_ollama_format",
            "test_invalid_structured_schema_is_rejected_before_loopback",
        )),
        ("API tests", api_tests, (
            "test_structured_inference_passes_schema_and_returns_validated_json",
            "test_structured_inference_rejects_unsupported_schema_before_runtime",
            "test_structured_inference_rejects_model_output_that_does_not_match_schema",
        )),
        ("CLI tests", cli_tests, (
            "test_infer_run_supports_json_schema_constrained_local_output",
            "test_infer_run_rejects_remote_reference_schemas_before_generation",
            "test_infer_run_rejects_symlinked_schema_files",
        )),
        ("browser acceptance", e2e_source, (
            "schema-constrained local inference", '"**/api/inference/generate/structured"',
            "SCHEMA VALIDATED", "JSON Schema for structured output",
        )),
        ("documentation", docs_source, (
            "POST /api/inference/generate/structured", "--schema-file",
            "512 requested output tokens", "additionalProperties",
        )),
    )
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                issues.append(f"Structured local inference {label} contract is missing: {token}")
    return issues


def runtime_trace_issues(trace_core: str, inference_api: str, diagnostics_api: str, client_source: str, app_source: str, panel_source: str, e2e_source: str) -> list[str]:
    """Check runtime request correlation, metadata-only storage, API controls and GUI wiring."""
    issues: list[str] = []
    for token in (
        "MAX_RUNTIME_TRACES = 200",
        "def record_runtime_trace(",
        '"inference.generate.structured"',
        "def list_runtime_traces(",
        "def clear_runtime_traces(",
        '"prompt_bodies_stored": False',
        '"response_bodies_stored": False',
        '"disk_persistence": False',
    ):
        if token not in trace_core:
            issues.append("Runtime trace store is missing its bounded privacy contract: " + token)
    for token in (
        'response.headers["X-Request-ID"] = request_id',
        'operation="inference.generate"',
        'operation="inference.generate.structured"',
        'operation="inference.benchmark"',
        'error_code="runtime_unavailable"',
    ):
        if token not in inference_api:
            issues.append("Inference API is missing request correlation or safe failure tracing: " + token)
    for token in (
        '@router.get("/runtime-traces")',
        '@router.delete("/runtime-traces")',
        "list_runtime_traces(limit, offset, search)",
        "clear_runtime_traces()",
    ):
        if token not in diagnostics_api:
            issues.append("Runtime trace history API is missing: " + token)
    for token in (
        "export type RuntimeTraceReport",
        "runtimeTraces: (limit = 50, offset = 0, search = \"\")",
        "clearRuntimeTraces:",
    ):
        if token not in client_source:
            issues.append("Typed runtime trace client is missing: " + token)
    if 'import RuntimeTracePanel from "./components/RuntimeTracePanel";' not in app_source or "<RuntimeTracePanel />" not in app_source:
        issues.append("Runtime page is not integrated with the request-trace panel.")
    for token in (
        'aria-label="Runtime request trace history"',
        "onClick={exportTraces}",
        "clearRuntimeTraces()",
        "axiom-runtime-traces.json",
        "Prompt text, generated responses, credentials",
    ):
        if token not in panel_source:
            issues.append("Runtime trace UI is missing export/clear/privacy behavior: " + token)
    for token in (
        "Runtime request traces show safe metadata, export JSON, and clear the in-memory buffer",
        "/api/diagnostics/runtime-traces",
        "axiom-runtime-traces.json",
        "Clear in-memory history",
    ):
        if token not in e2e_source:
            issues.append("Runtime trace browser acceptance coverage is missing: " + token)
    return issues


def engine_security_and_lifecycle_issues(engine_source: str, native_source: str, api_source: str, client_source: str, panel_source: str, e2e_source: str) -> list[str]:
    """Guard the native model execution path and trusted path/lifecycle controls."""
    issues: list[str] = []
    for token in (
        '_MODEL_ROOT_ENV = "AXIOM_MODEL_ROOTS"',
        "def _allowed_model_roots(",
        "def _path_is_within_root(",
        "path.relative_to(root)",
        "trusted-root escapes are not allowed",
        'details["model_directory_name"]',
        'details["adapter_directory_name"]',
        "def unload_local_model(",
        "_LOADED_RUNTIME: dict[str, Any] | None = None",
        "load_native_runtime(",
        '"attention_backend": model.last_attention_backend',
        '"prefill_ms": model.last_generation.get("prefill_ms", 0.0)',
        '"decode_ms": model.last_generation.get("decode_ms", 0.0)',
        '"time_to_first_token_ms": model.last_generation.get("time_to_first_token_ms", 0.0)',
        '"generation_tokens_per_second": model.last_generation.get("generation_tokens_per_second", 0.0)',
        "def discover_local_engine_assets(",
        "def check_local_adapter_compatibility(",
        "def validate_stop_sequences(",
        '"stop_sequences": list(stop_sequences)',
        '"stop_reason": stop_reason',
        "inspect_native_adapter_compatibility(",
        "cancel_event=cancel_event",
        "NativeEngineError",
    ):
        if token not in engine_source:
            issues.append("AXIOM Engine trusted path/native loader wiring is missing: " + token)
    for forbidden in ("Path(value)", "Path(model_path)", "Path(adapter_path)", "from transformers import", "AutoModelForCausalLM"):
        if forbidden in engine_source:
            issues.append("AXIOM Engine must not delegate inference or reevaluate raw path expressions: " + forbidden)
    for token in (
        "class NativeLlamaRuntime", "def forward(", "def generate(", "def _rms_norm(", "def _attention(", "scaled_dot_product_attention",
        "def _apply_rope(", "past_key_values", "repeat_interleave", "safe_open", "NativeGenerationCancelled", "cancel_event.is_set()",
        "model.safetensors", "NativeGenerationCancelled", "cancel_event.is_set()", "def inspect_native_adapter_compatibility(", "handle.get_slice(key).get_shape()", '"ROCm", rocm_available', '"Metal/MPS", mps_available',
        '"execution_backend": "pytorch-native-axiom"',
        '"transformers_generation_used": False', '"ollama_used": False',
    ):
        if token not in native_source:
            issues.append("AXIOM-owned native inference mechanics are missing: " + token)
    for forbidden in ("AutoModelForCausalLM.from_pretrained", "model.generate(**kwargs)", "requests.post("):
        if forbidden in native_source:
            issues.append("Native inference implementation delegates instead of executing locally: " + forbidden)
    for token in ("_validated_path(value, \"Base model\")", "_validated_path(value, \"LoRA adapter\")", "stop_sequences_must_be_bounded", '@router.get("/models")', "discover_local_engine_assets()", '@router.post("/generate/stream")', "generate_local_stream(", "watch_disconnect", "is_disconnected()", "Thread(", "Queue(maxsize=_STREAM_QUEUE_SIZE)", "cancel_event=cancel_event", "output.put(_STREAM_END, timeout=0.1)", '@router.post("/adapters/compatibility")', "check_local_adapter_compatibility(", '@router.post("/unload")', "unload_local_model()"):
        if token not in api_source:
            issues.append("AXIOM Engine API is missing trusted path validation or unload route: " + token)
    if "PurePath" in api_source or "paths_must_be_safe_relative" in api_source:
        issues.append("AXIOM Engine API has legacy path checks that conflict with the trusted-root resolver.")
    for token in ("axiomEngineModels:", '"/api/engine/models"', "checkAxiomEngineAdapterCompatibility:", '"/api/engine/adapters/compatibility"', "streamAxiomEngine:", 'api/engine/generate/stream', "unloadAxiomEngine:", '"/api/engine/unload"'):
        if token not in client_source:
            issues.append("Typed AXIOM Engine client is missing unload lifecycle support: " + token)
    for token in ("Select discovered native model", "axiomApi.axiomEngineModels()", "Check adapter compatibility", "axiomApi.checkAxiomEngineAdapterCompatibility(", "LoRA device compatibility report", "axiomApi.streamAxiomEngine(", "Stop generation", "Generation cancelled after", "AXIOM Native Engine prompt", "Stop sequences", "stopSequencesText", "stop_sequences: stopSequences", "Unload model", "axiomApi.unloadAxiomEngine()", "AXIOM_MODEL_ROOTS", "tokenizer.json", "safetensors"):
        if token not in panel_source:
            issues.append("AXIOM Native Engine GUI is missing native support/path safety controls: " + token)
    for token in ("then unloads it", "api/engine/models", 'selectOption("/models/llama-test")', 'requestBody.stop_sequences', "api/engine/adapters/compatibility", 'name: "Check adapter compatibility"', "LoRA device compatibility report", "api/engine/generate/stream", 'contentType: "application/x-ndjson"', "api/engine/unload", 'name: "Unload model"', 'name: "Stop generation"', "Generation cancelled after 0 token(s)"):
        if token not in e2e_source:
            issues.append("AXIOM Engine browser acceptance test is missing lifecycle coverage: " + token)
    return issues


def api_playground_issues(component: str, client: str) -> list[str]:
    """Validate the Playground's actual typed GET-only client contract and disclosure."""
    issues: list[str] = []
    client_is_bounded_get = all(
        token in client
        for token in (
            "PlaygroundEndpoint",
            "playgroundEndpointPaths",
            "playgroundRead",
            'method: "GET"',
        )
    )
    if "playgroundRead" not in component or not client_is_bounded_get:
        issues.append(
            "API playground must use the typed endpoint allowlist and GET-only API client."
        )
    if not all(
        token in component
        for token in ("allowlist", "GET only", "Arbitrary URLs", "mutating methods")
    ):
        issues.append(
            "API playground must disclose the allowlist, GET-only boundary, and rejected arbitrary/mutating requests."
        )
    return issues


def api_contract_examples_issues(
    core: str, api: str, server: str, cli: str, cli_main: str,
    client: str, component: str, e2e: str, tests: str, docs: str,
) -> list[str]:
    """Verify generated typed API examples stay tied to OpenAPI and secret-safe."""
    checks = (
        ("OpenAPI generator", core, (
            "def build_api_examples(", "MAX_OPERATIONS = 512", "MAX_EXAMPLES_BYTES",
            "def _safe_schema(", "def _example_value(", "def _typescript_type(",
            '"x-axiom-redacted"', "schema_defaults_replayed", "render_api_examples_markdown",
        )),
        ("API examples endpoint", api, (
            'APIRouter(prefix="/api/contract"', '@router.get("/examples")',
            "request.app.openapi()", "build_api_examples(",
        )),
        ("API server registration", server, ("api_examples_router",)),
        ("CLI examples command", cli, (
            '@api_app.command("examples")', '"--format"',
            "build_api_examples(", "render_api_examples_markdown",
        )),
        ("root CLI registration", cli_main, ('app.add_typer(api_app, name="api")',)),
        ("typed client", client, ("ApiContractExamples", "apiContractExamples:", '"/api/contract/examples"')),
        ("Control Center panel", component, (
            "Typed API examples", "axiomApi.apiContractExamples()",
            "selectedOperation.typescript", "selectedOperation.curl_template",
            "side_effecting", "never submitted",
        )),
        ("browser acceptance", e2e, (
            'getByLabel("Generated API operation")', "post_llm_chat",
            "$.body.api_key", "PostLlmChatRequest", "never submitted",
        )),
        ("generator regressions", tests, (
            "test_generated_examples_cover_the_live_openapi_operation_set",
            "test_secret_fields_are_redacted",
            "test_api_examples_endpoint_returns_generated_contract",
            "test_cli_emits_json_and_markdown",
        )),
        ("API docs", docs, (
            "GET /api/contract/examples", "axiom api",
            "synthetic", "variable names",
        )),
    )
    issues: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                issues.append(f"Typed API example {label} contract is missing: {token}")
    return issues


def hardware_snapshot_issues(
    core: str, api: str, server: str, cli_main: str, client: str,
    panel: str, app: str, e2e: str, tests: str, docs: str,
) -> list[str]:
    """Keep opt-in workspace hardware-history lifecycle wired across all surfaces."""
    checks = (
        ("snapshot store", core, (
            "class HardwareSnapshotStore", "MAX_SNAPSHOTS = 100",
            "MAX_RETENTION_DAYS = 3650", "def configure(", "def capture(", "def clear(",
            "HardwareSnapshotOptInRequired", "raise HardwareSnapshotOptInRequired(", "def _normalize_hardware(",
        )),
        ("snapshot API", api, (
            '"/hardware-snapshots"', '"/hardware-snapshots/settings"',
            "HardwareSnapshotOptInRequired", "status_code=201",
        )),
        ("server registration", server, ("hardware_snapshots_router",)),
        ("CLI lifecycle", cli_main, (
            'name="snapshots"', '@hardware_snapshots_app.command("enable")',
            '@hardware_snapshots_app.command("disable")',
            '@hardware_snapshots_app.command("capture")',
            '@hardware_snapshots_app.command("list")',
            '@hardware_snapshots_app.command("clear")',
        )),
        ("typed API client", client, (
            "HardwareSnapshotHistory", "hardwareSnapshots:",
            "updateHardwareSnapshotSettings:", "captureHardwareSnapshot:",
            "clearHardwareSnapshots:",
        )),
        ("Dashboard panel", panel, (
            "Hardware snapshot history", "Enable history", "Capture snapshot",
            "Clear history", "No automatic sampling",
        )),
        ("Dashboard registration", app, ("HardwareSnapshotPanel",)),
        ("browser acceptance", e2e, (
            "hardware history is opt-in", "captureCalls", "Enable history",
            "Capture snapshot", "Clear history",
        )),
        ("regression tests", tests, (
            "test_history_is_disabled_by_default", "test_explicit_capture_persists_only_allowlisted_hardware",
            "test_hardware_snapshot_api_requires_opt_in", "test_system_hardware_snapshot_commands",
        )),
        ("documentation", docs, (
            "GET /api/workspace/hardware-snapshots", "axiom system snapshots",
            "hardware-snapshots.json", "No background or periodic sampler",
        )),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Hardware snapshot {label} contract is missing: {token}")
    return failures


def resource_monitoring_issues(
    cli_source: str, panel_source: str, app_source: str, e2e_source: str,
    core_source: str, api_source: str, client_source: str,
    core_tests: str, api_tests: str,
) -> list[str]:
    """Require explicit opt-in, bounded persistent history and safe Runtime UI exports."""
    issues: list[str] = []
    groups = (
        ("CLI", cli_source, (
            'watch: bool = typer.Option(False, "--watch"',
            'samples: int = typer.Option(30, "--samples", min=1, max=300',
            'output.open("x"',
            "time.sleep(interval_seconds)",
            "--output requires explicit opt-in",
        )),
        ("Runtime GUI", panel_source, (
            "MAX_BROWSER_SAMPLES = 120",
            "if (!recording) return undefined;",
            "axiomApi.axiomEngineTelemetryHistory()",
            "axiomApi.configureAxiomEngineTelemetryHistory(",
            "axiomApi.recordAxiomEngineTelemetryHistory()",
            "axiomApi.clearAxiomEngineTelemetryHistory()",
            "persistentHistoryEnabled",
            "maxAgeDays",
            "window.setInterval",
            'anchor.download = "axiom-resource-telemetry.json"',
            "setSamples([])",
        )),
        ("Runtime page integration", app_source, (
            'import EngineTelemetryMonitor from "./components/EngineTelemetryMonitor";',
            "<EngineTelemetryMonitor />",
        )),
        ("Persistent history store", core_source, (
            "class ResourceTelemetryStore",
            "class ResourceTelemetryOptInRequired",
            "MAX_SAMPLES = 500",
            "DEFAULT_MAX_SAMPLES = 120",
            "DEFAULT_MAX_AGE_DAYS = 30",
            "def configure(",
            "def record(",
            "def clear(",
            "file_lock(",
            "atomic_write_text(",
            '"process_ids_included": False',
            '"prompt_or_generated_content_included": False',
            "0o600",
        )),
        ("Persistent history API", api_source, (
            'class ResourceTelemetryHistorySettingsRequest(BaseModel)',
            '@router.get("/telemetry-history")',
            '@router.put("/telemetry-history/settings")',
            '@router.post("/telemetry-history", status_code=201)',
            '@router.delete("/telemetry-history")',
            "ResourceTelemetryStore().configure(",
            "store.record(runtime_telemetry())",
        )),
        ("Typed API client", client_source, (
            "AxiomEngineResourceTelemetryHistory",
            "axiomEngineTelemetryHistory:",
            "configureAxiomEngineTelemetryHistory:",
            "recordAxiomEngineTelemetryHistory:",
            "clearAxiomEngineTelemetryHistory:",
        )),
        ("Core regression tests", core_tests, (
            "test_history_is_disabled_by_default_and_does_not_create_workspace_files",
            "test_record_persists_allowlisted_metrics_only_and_uses_private_file_modes",
            "test_sample_retention_is_bounded_by_count_and_age_and_disable_stops_new_records",
            "test_age_retention_prunes_old_records_and_clear_preserves_settings",
            "test_history_rejects_tampered_fingerprints_and_unsupported_fields",
        )),
        ("API regression tests", api_tests, (
            "test_resource_telemetry_api_has_explicit_opt_in_record_export_and_clear_lifecycle",
            "test_resource_telemetry_api_rejects_invalid_retention_settings_and_extra_fields",
        )),
        ("Browser acceptance", e2e_source, (
            "const baselineTelemetryCalls = telemetryCalls",
            "expect(telemetryCalls).toBe(baselineTelemetryCalls)",
            'name: "Start monitoring"',
            'name: "Stop monitoring"',
            "axiom-resource-telemetry.json",
            'name: "Clear history"',
            "privacy.model_paths_included",
            "Persist resource history in workspace",
            "Persistent history sample limit",
            "Persistent history retention days",
            "workspace-persistent",
            "persistedHistoryRecords",
        )),
    )
    for label, source, tokens in groups:
        for token in tokens:
            if token not in source:
                issues.append(f"{label} resource monitoring is missing: {token}")
    return issues


def benchmark_resource_trend_issues(panel: str, chart: str, e2e: str) -> list[str]:
    """Require benchmark-linked before/after samples, bounded trend charts and missing labels."""
    checks = (
        ("native benchmark panel", panel, (
            "axiomApi.axiomEngineTelemetry()", "telemetryBefore", "telemetryAfter",
            "setBenchmarkResourceSamples", "slice(-20)", "BenchmarkResourceTrends",
        )),
        ("resource trend charts", chart, (
            "System CPU after benchmark", "Process RSS change", "System RAM used change",
            "Device allocator change", "Not reported", "20 runs",
        )),
        ("browser acceptance", e2e, (
            "Benchmark resource trends", "1/20 runs", "System CPU after benchmark",
            "Run 1: Not reported", "Process RSS change", "Device allocator change",
        )),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Benchmark resource trend {label} is missing: {token}")
    return failures


def telemetry_export_privacy_issues(monitor: str, e2e: str, docs: str) -> list[str]:
    """Require export formats, explicit field privacy controls and deterministic clear."""
    checks = (
        ("monitor UI", monitor, (
            "Export JSON", "Export CSV", 'aria-label="Telemetry export privacy controls"',
            "includeCpu", "includeMemory", "includeProcess", "includeDevice",
            "anchor.download = \"axiom-resource-telemetry.csv\"",
            "process_ids_included: false", "setSamples([])",
        )),
        ("browser acceptance", e2e, (
            "Include process RSS/thread metrics", "axiom-resource-telemetry.csv",
            "process_metrics_included", "process_ids_included", "system_cpu_percent",
            "Clear history",
        )),
        ("documentation", docs, (
            "JSON and CSV", "privacy controls", "Process IDs are never exported",
            "browser-memory-only", "workspace-persistent",
        )),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Telemetry export/privacy {label} is missing: {token}")
    return failures


def persisted_schema_inventory_issues(docs: str, sources: dict[str, str]) -> list[str]:
    """Ensure every persisted store and its real migration/rollback rules stay documented."""
    expected_doc_tokens = (
        ".axiom/workflows.json", "schema_version: 2", "Schema-v1 records are accepted",
        ".axiom/experiments.json", ".axiom/evaluations.json", ".axiom/agent-profiles.json",
        ".axiom/models.json", ".axiom/integrations.json", ".axiom/tool_permissions.json",
        ".axiom/security-events.json", ".axiom/llm-provider-profiles.json",
        ".axiom/hardware-snapshots.json", ".axiom/resource-telemetry.json", "~/.axiom/workspaces.json",
        ".axiom/prediction_jobs.json", "Fine-tuning jobs", "Model Studio jobs",
        "manifest.json", "Runtime traces are in-memory only", "browser-memory-only",
        "Model Studio's reader does not yet reject every future version explicitly", "default-deny",
    )
    failures = [
        f"Persisted-schema documentation is missing: {token}"
        for token in expected_doc_tokens if token not in docs
    ]
    # A version token elsewhere in the document must not mask a stale row for a
    # particular store (for example, workflows v1 while LLM profiles remain v2).
    versioned_inventory_rows = (
        ("workflows", ".axiom/workflows.json", "schema_version: 2"),
        ("security_events", ".axiom/security-events.json", "schema_version: 2"),
        ("llm_profiles", ".axiom/llm-provider-profiles.json", "schema_version: 2"),
        ("hardware_snapshots", ".axiom/hardware-snapshots.json", "schema_version: 1"),
        ("resource_telemetry", ".axiom/resource-telemetry.json", "schema_version: 1"),
        ("workspaces", "~/.axiom/workspaces.json", "schema_version: 1"),
        ("prediction_jobs", ".axiom/prediction_jobs.json", "schema_version: 1"),
    )
    for name, store_path, expected_version in versioned_inventory_rows:
        row = next(
            (line for line in docs.splitlines() if line.startswith("|") and store_path in line),
            "",
        )
        if not row:
            failures.append(f"Persisted-schema documentation is missing the {name} inventory row: {store_path}")
        elif expected_version not in row:
            failures.append(
                f"Persisted-schema documentation for {name} has a stale schema version; "
                f"expected {expected_version} in the {store_path} inventory row."
            )
    code_contracts = {
        "workflows": ("WORKFLOW_SCHEMA_VERSION = 2",),
        "llm_profiles": ("SCHEMA_VERSION = 2", "version not in {1, SCHEMA_VERSION}"),
        "security_events": ("LEGACY_SCHEMA_VERSION = 1", "SCHEMA_VERSION = 2"),
        "hardware_snapshots": ("SCHEMA_VERSION = 1",),
        "resource_telemetry": ("SCHEMA_VERSION = 1", "MAX_SAMPLES = 500", "DEFAULT_MAX_SAMPLES = 120"),
        "workspaces": ("_SCHEMA_VERSION = 1",),
        "prediction_jobs": ('"schema_version": 1', 'payload["schema_version"] != 1'),
        "integrations": ('data.get("trust_status", "unverified")', 'data.get("connection_status", "not_inspected")'),
        "permissions": ('"active_profile"', '"profiles"', "_default_config()"),
        "registry": ("legacy_resource_id(", 'registry_file = self.root / "models.json"'),
        "workspace_metadata": ("ARCHIVE_SCHEMA_VERSION = 1", '"agent_profiles": "agent-profiles.json"'),
        "fine_tuning": ('"schema_version": 1', "def _read_job("),
        "model_studio": ('"schema_version": 1', "def _read_job("),
        "runtime_traces": ("MAX_RUNTIME_TRACES", "_traces_by_workspace"),
    }
    for name, tokens in code_contracts.items():
        source = sources.get(name, "")
        if not source:
            failures.append(f"Persisted-schema source inventory is missing: {name}")
            continue
        for token in tokens:
            if token not in source:
                failures.append(f"Persisted-schema documentation/source contract for {name} is missing: {token}")
    return failures


def supply_chain_issues(
    core: str, api: str, cli: str, client: str, gui: str,
    tests: str, api_tests: str, e2e: str, docs: str,
) -> list[str]:
    """Require SPDX inventory, explicitly authorized online scanning, and regression coverage."""
    checks = (
        ("SPDX inventory", core, (
            "def generate_spdx_sbom(", '"SPDX-2.3"', '"PACKAGE-MANAGER"',
            '"DEPENDS_ON"', "MAX_DISTRIBUTIONS = 4096", "MAX_TOTAL_REQUIREMENTS = 50_000",
            "content_sha256=", "network_access_performed=false", "host_paths_collected=false",
            "host_paths_collected", "network_access_performed",
        )),
        ("Dependency audit", core, (
            "def audit_dependency_vulnerabilities(", "allow_network: bool",
            "explicit network authorization", "MAX_AUDIT_OUTPUT_BYTES",
            '"--local"', '"--format", "json"', "subprocess.TimeoutExpired",
            "PIP_INDEX_URL", '"unavailable"',
        )),
        ("Security API", api, (
            '@router.get("/sbom")', '@router.post("/dependency-audit")',
            "allow_network=request.allow_network", "timeout_seconds=request.timeout_seconds",
        )),
        ("Security CLI", cli, (
            '@security_app.command("sbom")', '@security_app.command("vulnerabilities")',
            '"--allow-network"', "write_report_new(",
        )),
        ("Typed client", client, (
            "SoftwareBillOfMaterials", "DependencyVulnerabilityReport",
            "securitySoftwareBillOfMaterials:", "scanDependencyVulnerabilities:",
        )),
        ("Diagnostics GUI", gui, (
            "exportSoftwareBillOfMaterials", "scanDependencyVulnerabilities",
            "window.confirm(", "Export SPDX 2.3 SBOM", "Scan dependencies (network)",
            "unavailable scanner is never described as a clean scan",
        )),
        ("Core regression tests", tests, (
            "test_spdx_sbom_contains_package_urls_versions_and_installed_dependency_relationships",
            "test_spdx_sbom_validates_against_the_official_23_json_schema",
            "test_spdx_sbom_bounds_aggregate_requirement_metadata",
            "test_spdx_sbom_rejects_unbounded_or_invalid_installed_metadata",
            "test_pip_audit_parser_rejects_exit_status_and_finding_mismatches",
            "test_report_export_rejects_symlinked_parent_and_existing_symlink",
            "test_vulnerability_scan_requires_explicit_network_authorization",
            "test_vulnerability_scan_reports_unavailable_scanner_without_claiming_clean",
            "test_vulnerability_scan_invokes_scanner_only_after_authorization_and_drops_secret_environment",
            "test_vulnerability_scan_does_not_report_clean_on_timeout",
        )),
        ("API regression tests", api_tests, (
            "test_security_supply_chain_api_returns_spdx_sbom",
            "test_security_supply_chain_api_requires_explicit_network_authorization",
            "test_security_supply_chain_api_runs_only_authorized_bounded_dependency_scan",
        )),
        ("Browser acceptance", e2e, (
            "Export SPDX 2.3 SBOM", "Scan dependencies (network)",
            'page.waitForEvent("dialog")', "axiom-installed-environment.spdx.json",
            "axiom-dependency-vulnerability-report.json", "allow_network: true",
        )),
        ("Security documentation and optional scanner", docs, (
            "SPDX 2.3", "pip-audit", "network authorization",
            "unavailable", "installed Python", "security-audit", "pip-audit>=2.10.1,<3",
        )),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Software supply-chain {label} is missing: {token}")
    return failures


def dataset_snapshot_issues(
    core: str, api: str, cli: str, client: str, gui: str,
    core_tests: str, api_tests: str, cli_tests: str, e2e: str, docs: str,
) -> list[str]:
    """Require immutable content-addressed storage, safe workspace routes and lineage acceptance."""
    checks = (
        ("Snapshot store", core, (
            "class DatasetSnapshotStore",
            "MAX_SNAPSHOTS = 250",
            "MAX_DATASET_BYTES = 1024 * 1024 * 1024",
            "MAX_TOTAL_SNAPSHOT_BYTES = 10 * 1024 * 1024 * 1024",
            "def create_snapshot(",
            "def verify(",
            "file_lock(",
            "atomic_write_text(",
            "record_sha256",
            "parent_manifest_sha256",
            "object_name",
        )),
        ("Workspace API", api, (
            "class WorkspaceDatasetSnapshotInput(BaseModel)",
            '@router.get("/dataset-snapshots")',
            '@router.post("/dataset-snapshots", status_code=201)',
            '@router.get("/dataset-snapshots/{snapshot_id}/verify")',
            "DatasetSnapshotStore(active_workspace_root())",
            "open_regular_file(candidate)",
        )),
        ("Dataset CLI", cli, (
            '@dataset_app.command("snapshot")',
            '@dataset_app.command("snapshots")',
            '@dataset_app.command("snapshot-verify")',
            "DatasetSnapshotStore(workspace)",
            "find_existing_relative_path(data_root",
        )),
        ("Typed client", client, (
            "DatasetSnapshotManifest",
            "DatasetSnapshotHistory",
            "DatasetSnapshotVerification",
            "datasetSnapshots:",
            "createDatasetSnapshot:",
            "verifyDatasetSnapshot:",
        )),
        ("Datasets GUI", gui, (
            "snapshotHistory",
            "snapshotParentId",
            "runSnapshot",
            "verifySnapshot",
            "createDatasetSnapshot(",
            "verifyDatasetSnapshot(",
            "Parent dataset snapshot",
            "Immutable snapshot history",
            "Snapshot",
        )),
        ("Store regression tests", core_tests, (
            "test_snapshot_stores_content_addressed_read_only_object_and_value_free_manifest",
            "test_identical_content_is_deduplicated_and_parent_lineage_is_verifiable",
            "test_parent_must_be_a_valid_earlier_snapshot_in_the_same_workspace",
            "test_snapshot_enforces_source_type_format_and_size_bounds",
            "test_snapshot_rejects_tampered_manifest_index_and_hash_chain",
            "test_snapshot_verification_detects_modified_object_bytes",
            "test_snapshot_rejects_symlinked_workspace_metadata",
        )),
        ("API regression tests", api_tests, (
            "test_dataset_snapshot_api_creates_lists_and_verifies_workspace_snapshots",
            "test_dataset_snapshot_api_rejects_unsafe_sources_parent_and_byte_overflow",
            "test_dataset_snapshot_api_links_parent_manifest_to_prior_snapshot",
            "test_dataset_snapshot_api_returns_not_found_for_unknown_snapshot",
        )),
        ("CLI regression tests", cli_tests, (
            "test_dataset_snapshot_cli_creates_lists_and_verifies_immutable_history",
            "test_dataset_snapshot_cli_rejects_files_outside_workspace_data",
        )),
        ("Browser acceptance", e2e, (
            "Datasets creates immutable content-addressed snapshots",
            "Parent dataset snapshot",
            "Verify",
            "parent_snapshot_id: firstId",
            "Snapshot content hash and lineage verified.",
        )),
        ("Snapshot documentation", docs, (
            "content-addressed",
            "hash-chained",
            "read-only",
            "parent_snapshot_id",
            "not a signed",
            "axiom dataset snapshot",
        )),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Dataset snapshot {label} is missing: {token}")
    return failures


def package_verification_issues(
    verifier: str, workflow: str, tests: str, docs: str,
) -> list[str]:
    """Require wheel+sdist construction, safe archive inspection and import testing on every OS."""
    checks = (
        ("artifact verifier", verifier, (
            "def verify_package_artifacts(", "MAX_ARCHIVE_MEMBERS",
            "MAX_MEMBER_BYTES", "MAX_TOTAL_BYTES", "def _verify_wheel(",
            "def _verify_sdist(", "EXPECTED_ENTRY_POINT", "axiom/webui/index.html",
            "entry_points.txt", "testzip()", "tarfile.open", "install_wheel",
        )),
        ("cross-platform workflow", workflow, (
            "matrix.os", "node-version: \"22\"", "npm ci", "npm run build",
            "python -m build --sdist --wheel --outdir dist",
            "python scripts/verify_package_artifacts.py --dist dist --install-wheel --json",
            "tests/test_package_artifact_verification.py", "tests/test_persisted_schema_inventory.py", "tests/test_dataset_snapshots.py", "tests/test_dataset_snapshots_api.py", "tests/test_dataset_snapshots_cli.py",
        )),
        ("archive regression tests", tests, (
            "test_verifies_wheel_and_sdist_bundle_cli_and_control_center",
            "test_rejects_wheel_path_traversal_and_special_files",
            "test_rejects_wheel_symlink_members",
            "test_rejects_case_colliding_wheel_paths",
            "test_rejects_wrong_cli_entry_point",
            "test_rejects_sdist_symbolic_links",
            "test_rejects_wheel_missing_control_center_css_asset",
            "test_rejects_sdist_missing_control_center_css_asset",
        )),
        ("packaging documentation", docs, (
            "Ubuntu", "Windows", "macOS", "wheel", "sdist",
            "axiom/webui", "without extracting",
        )),
    )
    failures: list[str] = []
    for label, source, tokens in checks:
        for token in tokens:
            if token not in source:
                failures.append(f"Cross-platform packaging {label} is missing: {token}")
    return failures


def inspect(root: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    required = [
        "pyproject.toml", "axiom/version.py", "ROADMAP.md", "src/App.tsx",
        "src/api/client.ts", "src/components/ApiPlayground.tsx", "src/components/RuntimeConnectorPanel.tsx", "src/components/AgentProfiles.tsx", "src/components/LocalInferencePanel.tsx", "src/components/RuntimeTracePanel.tsx", "src/components/EngineTelemetryMonitor.tsx", "src/components/LocalAxiomEnginePanel.tsx", "src/components/BenchmarkResourceTrends.tsx", "src/components/ModelStudioPanel.tsx", "src/styles.css",
        "axiom/core/json_schema.py", "tests/test_json_schema.py", "tests/test_inference_structured_api.py", "docs/wiki/Structured-Output.md",
                "axiom/api/server.py", "axiom/api/workspace.py", "axiom/datasets/snapshots.py", "axiom/api/workflows.py", "axiom/api/engine.py", "axiom/api/inference.py", "axiom/api/model_studio.py", "axiom/core/inference.py", "axiom/core/workflows.py", "axiom/core/ollama_service.py", "axiom/core/native_engine.py", "axiom/core/model_studio.py", "axiom/core/runtime_traces.py", "axiom/core/resource_telemetry.py", "axiom/models/compatibility.py", "axiom/api/diagnostics.py", "axiom/api/runtime_discovery.py", "axiom/api/agent_profiles.py", "axiom/core/runtime_discovery.py", "axiom/core/agent_profiles.py", "axiom/core/engine_fabric_profiles.py", "axiom/cli/agent_profiles.py", "axiom/cli/workflows.py", "axiom/cli/engine.py", "axiom/cli/model_studio.py", "scripts/release_smoke.py", "scripts/verify_package_artifacts.py", "tests/test_package_artifact_verification.py", "tests/test_persisted_schema_inventory.py",
        "tests/test_release_smoke.py", "tests/test_ollama_service.py", "tests/test_ollama_service_api.py", "tests/test_ollama_model_operations.py", "tests/test_cli_inference.py", "tests/test_axiom_engine.py", "tests/test_axiom_engine_api.py", "axiom/core/benchmark_profiles.py", "tests/test_benchmark_profiles.py", "tests/test_benchmark_profiles_api.py", "tests/test_axiom_engine_cli.py", "tests/test_native_engine.py", "tests/test_model_studio.py", "tests/test_engine_runtime_status.py", "tests/test_cli_model_studio_jobs.py", "e2e/axiom-engine.spec.ts", "e2e/helpers/auth.ts", "e2e/modules.spec.ts", "e2e/workflows.spec.ts", "e2e/evaluation.spec.ts",
        "e2e/model-registry-filters.spec.ts", "e2e/ollama-service.spec.ts", "e2e/inference.spec.ts", "e2e/inference-benchmark.spec.ts", "e2e/runtime-traces.spec.ts", "e2e/telemetry-monitor.spec.ts", "e2e/dataset-snapshots.spec.ts", "e2e/axiom-engine.spec.ts", "tests/test_workspace_model_registry_filters.py", "tests/test_workflow_store_pagination.py", "tests/test_workflow_api_pagination.py", "tests/test_workflow_cli.py",
        "tests/test_model_compatibility.py", "tests/test_inference_benchmark.py", "tests/test_inference_benchmark_api.py", "tests/test_runtime_traces.py", "tests/test_runtime_traces_api.py", "tests/test_resource_telemetry.py", "tests/test_resource_telemetry_api.py", "tests/test_runtime_discovery.py", "tests/test_runtime_discovery_api.py", "tests/test_agent_profiles.py", "tests/test_agent_profiles_api.py", "tests/test_agent_profiles_cli.py", "tests/test_engine_fabric_profiles.py", "tests/test_engine_fabric_profiles_api.py", "tests/test_engine_fabric_profiles_cli.py", "e2e/api-playground.spec.ts", "e2e/runtime-connectors.spec.ts", "e2e/agent-profiles.spec.ts", "e2e/engine-fabric.spec.ts",
        "docs/Runtime-Discovery.md", "docs/Agent-Profiles.md", "docs/Api-Playground.md", "docs/Local-Inference.md", "docs/Inference-Benchmarking.md", "docs/Local-Engine.md", "docs/Engine-Fabric.md",
        "docs/wiki/Security-and-Privacy.md", "docs/Dataset-Snapshots.md", "docs/Packaging.md", "docs/Storage-Schemas-and-Migrations.md", "docs/DATA-RETENTION.md", ".github/workflows/quality-gates.yml", ".github/workflows/nightly-build.yml",
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
    client = (root / "src/api/client.ts").read_text(encoding="utf-8")
    failures.extend(page_registration_issues(app_source, "api-playground"))
    failures.extend(page_registration_issues(app_source, "agent-profiles"))
    failures.extend(module_catalog_issues(app_source))

    snapshot_core = (root / "axiom/datasets/snapshots.py").read_text(encoding="utf-8")
    snapshot_api = workspace_api = (root / "axiom/api/workspace.py").read_text(encoding="utf-8")
    snapshot_cli = (root / "axiom/cli/dataset_profile.py").read_text(encoding="utf-8")
    snapshot_client = client
    snapshot_core_tests = (root / "tests/test_dataset_snapshots.py").read_text(encoding="utf-8")
    snapshot_api_tests = (root / "tests/test_dataset_snapshots_api.py").read_text(encoding="utf-8")
    snapshot_cli_tests = (root / "tests/test_dataset_snapshots_cli.py").read_text(encoding="utf-8")
    snapshot_e2e = (root / "e2e/dataset-snapshots.spec.ts").read_text(encoding="utf-8")
    snapshot_docs = (root / "docs/Dataset-Snapshots.md").read_text(encoding="utf-8")
    failures.extend(dataset_snapshot_issues(
        snapshot_core, snapshot_api, snapshot_cli, snapshot_client, app_source,
        snapshot_core_tests, snapshot_api_tests, snapshot_cli_tests, snapshot_e2e, snapshot_docs,
    ))
    component = (root / "src/components/ApiPlayground.tsx").read_text(encoding="utf-8")

    supply_chain_core = (root / "axiom/core/supply_chain.py").read_text(encoding="utf-8")
    supply_chain_api = (root / "axiom/api/security.py").read_text(encoding="utf-8")
    supply_chain_cli = (root / "axiom/cli/extended.py").read_text(encoding="utf-8")
    supply_chain_e2e = (root / "e2e/security-supply-chain.spec.ts").read_text(encoding="utf-8")
    supply_chain_tests = (root / "tests/test_supply_chain.py").read_text(encoding="utf-8")
    supply_chain_api_tests = (root / "tests/test_security_supply_chain_api.py").read_text(encoding="utf-8")
    supply_chain_docs = (root / "docs/wiki/Security-and-Privacy.md").read_text(encoding="utf-8")
    supply_chain_docs += "\\n" + (root / "pyproject.toml").read_text(encoding="utf-8")
    failures.extend(supply_chain_issues(
        supply_chain_core, supply_chain_api, supply_chain_cli, client,
        app_source, supply_chain_tests, supply_chain_api_tests,
        supply_chain_e2e, supply_chain_docs,
    ))
    workspace_api = (root / "axiom/api/workspace.py").read_text(encoding="utf-8")
    model_registry_e2e = (root / "e2e/model-registry-filters.spec.ts").read_text(encoding="utf-8")
    failures.extend(model_registry_filter_issues(workspace_api, client, app_source, model_registry_e2e))
    compatibility_core = (root / "axiom/models/compatibility.py").read_text(encoding="utf-8")
    compatibility_e2e = model_registry_e2e
    failures.extend(model_compatibility_issues(compatibility_core, workspace_api, client, app_source, compatibility_e2e))
    inference_core = (root / "axiom/core/inference.py").read_text(encoding="utf-8")
    inference_api = (root / "axiom/api/inference.py").read_text(encoding="utf-8")
    inference_panel = (root / "src/components/LocalInferencePanel.tsx").read_text(encoding="utf-8")
    inference_e2e = (root / "e2e/inference-benchmark.spec.ts").read_text(encoding="utf-8")
    failures.extend(inference_benchmark_issues(inference_core, inference_api, client, inference_panel, inference_e2e))
    ollama_service = (root / "axiom/core/ollama_service.py").read_text(encoding="utf-8")
    ollama_cli = (root / "axiom/cli/inference.py").read_text(encoding="utf-8")
    ollama_e2e = (root / "e2e/ollama-service.spec.ts").read_text(encoding="utf-8")
    ollama_tests = (root / "tests/test_ollama_service.py").read_text(encoding="utf-8")
    inference_core_tests = (root / "tests/test_inference_core.py").read_text(encoding="utf-8")
    ollama_model_tests = (root / "tests/test_ollama_model_operations.py").read_text(encoding="utf-8")
    ollama_api_tests = (root / "tests/test_ollama_service_api.py").read_text(encoding="utf-8")
    for token in ("OLLAMA_HOST = \"127.0.0.1:11434\"", "def start_service(", "def stop_service(", "def service_status(", "def start_model_pull(", "def cancel_model_pull(", "_MAX_CONCURRENT_PULLS = 2", "_MODEL_PULL_TIMEOUT_SECONDS = 1800.0", "def _matches_state("):
        if token not in ollama_service:
            failures.append("Managed Ollama service is missing a safety/lifecycle contract: " + token)
    for token in ("def _sampling_kwargs(", "top_p: float | None", '@router.post("/generate/stream")', '@router.get("/models/running")', '@router.post("/service/start")', '@router.post("/service/stop")', '@router.post("/models/pull")', '@router.post("/models/inspect")', '@router.post("/models/unload")', "AXIOM_ALLOW_REMOTE_RUNTIME_MANAGEMENT", "cancel_model_pull("):
        if token not in inference_api:
            failures.append("Ollama API is missing local lifecycle/model operations: " + token)
    for token in ("def validate_sampling_options(", "def stream_local_inference(", "def list_running_local_models(", "def inspect_local_model(", "def unload_local_model(", '"keep_alive": 0', '"template_configured": bool'):
        if token not in inference_core:
            failures.append("Ollama model operations are missing: " + token)
    for token in ("export type LocalInferenceInput", "stop_sequences?: string[]", "streamLocalInference:", "LocalInferenceStreamEvent", "runningOllamaModels:", "startLocalOllama:", "stopLocalOllama:", "startOllamaModelPull:", "cancelOllamaModelPull:", "inspectOllamaModel:", "unloadOllamaModel:", "export type OllamaModelDetails"):
        if token not in client:
            failures.append("Typed Ollama client is missing a local model operation: " + token)
    for token in ("Advanced sampling options", "top_p: topPValue", "min_p: minPValue", "stop_sequences: stops", "Stop generation", "streamLocalInference(", "Start Ollama", "Stop AXIOM service", "Pull model", "Show loaded models", "Cancel download", "Inspect model", "Unload from memory", "Local model metadata"):
        if token not in inference_panel:
            failures.append("Runtime GUI is missing Ollama service/model controls: " + token)
    for token in ("--top-p", "--top-k", "--min-p", "--repeat-penalty", "--num-ctx", "--stop", '@infer_app.command("ps")', '@service_app.command("start")', '@service_app.command("stop")', '@infer_app.command("pull")', '@infer_app.command("inspect")', '@infer_app.command("unload")'):
        if token not in ollama_cli:
            failures.append("Ollama CLI is missing a lifecycle/model command: " + token)
    for token in ("test_service_status_exposes_control_capabilities", "test_stop_refuses_to_signal_a_process_not_owned_by_axiom", "test_start_terminates_process_if_ownership_state_cannot_be_saved", "test_start_reaps_process_if_process_identity_cannot_be_verified", "test_progress_output_removes_terminal_controls"):
        if token not in ollama_tests:
            failures.append("Ollama process-safety tests are missing: " + token)
    for token in ("test_stream_local_inference_yields_token_events", "test_running_model_inventory_reports_loaded_memory_without_raw_content", "test_model_inspection_returns_metadata_without_model_templates", "test_unload_releases_memory_without_deleting_downloaded_model"):
        if token not in ollama_model_tests:
            failures.append("Ollama metadata/unload tests are missing: " + token)
    for token in ("test_advanced_sampling_options_are_forwarded", "test_invalid_sampling_options_are_rejected"):
        if token not in inference_core_tests:
            failures.append("Ollama advanced sampling core tests are missing: " + token)
    schema_core = (root / "axiom/core/json_schema.py").read_text(encoding="utf-8")
    schema_tests = (root / "tests/test_json_schema.py").read_text(encoding="utf-8")
    structured_api_tests = (root / "tests/test_inference_structured_api.py").read_text(encoding="utf-8")
    structured_docs = (root / "docs/wiki/Structured-Output.md").read_text(encoding="utf-8")
    structured_docs += "\\n" + (root / "docs/Local-Inference.md").read_text(encoding="utf-8")
    failures.extend(structured_inference_issues(
        schema_core, inference_core, inference_api, client, ollama_cli,
        inference_panel, schema_tests, inference_core_tests, structured_api_tests,
        (root / "tests/test_cli_inference.py").read_text(encoding="utf-8"),
        (root / "e2e/inference.spec.ts").read_text(encoding="utf-8"), structured_docs,
    ))
    inference_stream_e2e = (root / "e2e/inference.spec.ts").read_text(encoding="utf-8")
    for token in ("test_streaming_api_passes_advanced_sampling_options", "test_generate_api_rejects_oversized_context_and_stop_sequence_lists", "test_streaming_inference_emits_ndjson_token_deltas", "test_streaming_inference_sanitizes_runtime_failures"):
        if token not in ollama_api_tests:
            failures.append("Ollama streaming API coverage is missing: " + token)
    for token in ('"/api/inference/generate/stream"', "top_p: 0.83", "min_p: 0.2", 'stop_sequences: ["DONE", "END"]', "Stop generation", "Hello from the mocked local model"):
        if token not in inference_stream_e2e:
            failures.append("Streaming generation browser acceptance coverage is missing: " + token)
    for token in ("start Ollama, pull a model with progress", "Show loaded models", "Inspect model", "Unload from memory", '"/api/inference/models/running"', '"/api/inference/models/inspect"', '"/api/inference/models/unload"'):
        if token not in ollama_e2e:
            failures.append("Ollama browser acceptance coverage is missing: " + token)
    trace_core = (root / "axiom/core/runtime_traces.py").read_text(encoding="utf-8")
    diagnostics_api = (root / "axiom/api/diagnostics.py").read_text(encoding="utf-8")
    trace_panel = (root / "src/components/RuntimeTracePanel.tsx").read_text(encoding="utf-8")
    trace_e2e = (root / "e2e/runtime-traces.spec.ts").read_text(encoding="utf-8")
    failures.extend(runtime_trace_issues(trace_core, inference_api, diagnostics_api, client, app_source, trace_panel, trace_e2e))
    engine_source = (root / "axiom/core/engine.py").read_text(encoding="utf-8")
    native_source = (root / "axiom/core/native_engine.py").read_text(encoding="utf-8")
    engine_api = (root / "axiom/api/engine.py").read_text(encoding="utf-8")
    engine_panel = (root / "src/components/LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    engine_e2e = (root / "e2e/axiom-engine.spec.ts").read_text(encoding="utf-8")
    engine_cli = (root / "axiom/cli/engine.py").read_text(encoding="utf-8")
    benchmark_profiles_core = (root / "axiom/core/benchmark_profiles.py").read_text(encoding="utf-8")
    benchmark_profiles_api_tests = (root / "tests/test_benchmark_profiles_api.py").read_text(encoding="utf-8")
    benchmark_profiles_cli_tests = (root / "tests/test_axiom_engine_cli.py").read_text(encoding="utf-8")
    benchmark_profiles_core_tests = (root / "tests/test_benchmark_profiles.py").read_text(encoding="utf-8")
    benchmark_profiles_docs = (root / "docs/Inference-Benchmarking.md").read_text(encoding="utf-8")
    failures.extend(benchmark_profile_issues(
        benchmark_profiles_core, engine_api, engine_cli, client, engine_panel,
        benchmark_profiles_core_tests, benchmark_profiles_api_tests,
        benchmark_profiles_cli_tests, engine_e2e, benchmark_profiles_docs,
    ))
    failures.extend(engine_security_and_lifecycle_issues(engine_source, native_source, engine_api, client, engine_panel, engine_e2e))
    native_model_studio = (root / "axiom/core/model_studio.py").read_text(encoding="utf-8")
    native_model_studio_api = (root / "axiom/api/model_studio.py").read_text(encoding="utf-8")
    native_model_studio_cli = (root / "axiom/cli/model_studio.py").read_text(encoding="utf-8")
    native_model_studio_panel = (root / "src/components/ModelStudioPanel.tsx").read_text(encoding="utf-8")
    native_engine_panel = (root / "src/components/LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    native_engine_cli = (root / "axiom/cli/engine.py").read_text(encoding="utf-8")
    native_engine_docs = (root / "docs/Local-Engine.md").read_text(encoding="utf-8")
    native_model_studio_tests = (root / "tests/test_model_studio.py").read_text(encoding="utf-8")
    native_model_studio_cli_tests = (root / "tests/test_cli_model_studio_jobs.py").read_text(encoding="utf-8")
    native_engine_api_tests = (root / "tests/test_axiom_engine_api.py").read_text(encoding="utf-8")
    native_engine_tests = (root / "tests/test_native_engine.py").read_text(encoding="utf-8")
    native_runtime_status_tests = (root / "tests/test_engine_runtime_status.py").read_text(encoding="utf-8")
    native_engine_cli_tests = (root / "tests/test_axiom_engine_cli.py").read_text(encoding="utf-8")
    native_engine_e2e = (root / "e2e/axiom-engine.spec.ts").read_text(encoding="utf-8")
    native_engine_cli = (root / "axiom/cli/engine.py").read_text(encoding="utf-8")
    native_model_studio_e2e = (root / "e2e/model-studio.spec.ts").read_text(encoding="utf-8")

    native_readiness_contracts = (
        ("dependency readiness", engine_source, (
            'minimum_versions = {"torch": "2.2.0", "tokenizers": "0.19.0", "safetensors": "0.4.0"}',
            '"ready": ready', '"packages": packages', '"versions": versions', '"minimum_versions": minimum_versions',
            '"architecture": _LOADED_ARCHITECTURE', '"loaded_device":', '"devices": devices',
            '"recommended_device": recommended_device', '"remote_code": False',
            '"supported_model_types": ["llama", "mistral", "qwen2", "qwen3"]',
        )),
        ("explicit device selection", engine_source, (
            'choices = {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}',
            "def _resolve_execution_device(", "torch_name = \"cuda\" if selected[\"id\"] == \"rocm\" else selected[\"id\"]",
        )),
        ("staged public native model acquisition", native_model_studio, (
            "def pull_native_model(", 'staging = root / f".{model_name}.', "os.replace(staging, target)",
            "on_progress=on_progress", "cancel_event=cancel_event", '"remote_code_executed": False',
            "NativeLlamaConfig.from_mapping(config)", "_MAX_NATIVE_MODEL_BYTES",
        )),
        ("Model Studio API", native_model_studio_api, (
            'Literal["pull", "pull_native", "install_gguf", "convert_gguf"]',
            '@router.post("/jobs", status_code=202)', "studio.start_job(request.model_dump())",
            '@router.post("/jobs/{job_id}/cancel")',
        )),
        ("Model Studio CLI", native_model_studio_cli, (
            '@studio_app.command("pull-native")', "studio.pull_native_model(",
            "cancellable_byte_progress", "cancel_event=cancel_event", "on_progress=progress",
        )),
        ("Model Studio GUI", native_model_studio_panel, (
            "Pull a model for AXIOM Native Engine", 'operation:"pull_native"',
            "Pull native checkpoint", "job.progress_bytes", "Cancel operation",
        )),
        ("native Engine CLI surface", native_engine_cli, (
            'def engine_pull(', 'device: str = typer.Option("auto", "--device"',
        )),
        ("native API device dispatch", engine_api, (
            'device: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")',
            "device_preference=request.device",
        )),
        ("native API device and benchmark-cancellation regression", native_engine_api_tests, (
            "test_engine_stream_endpoint_yields_token_snapshots_and_final_metrics",
            'captured["device_preference"] == "cuda"',
            "test_native_benchmark_api_signals_worker_when_client_disconnects",
        )),
        ("native CLI device/readiness regression", native_engine_cli_tests, (
            "test_engine_run_passes_explicit_device_to_native_stream",
            "test_engine_status_json_reports_readiness_devices_and_local_only_scope",
        )),
        ("native Runtime GUI", native_engine_panel, (
            'aria-label="Execution device"', "status?.devices", "status?.recommended_device",
            "GENERATED ON THIS MACHINE", "LOCAL ONLY",
        )),
        ("native model and execution docs", native_engine_docs, (
            "stages downloads before publishing", "public Llama/Mistral/Qwen2/Qwen3 safetensors",
            "CPU, NVIDIA CUDA, AMD ROCm, Apple Metal/MPS, and Intel XPU",
            "remote model code", "does not silently route generation through another engine",
            "### Hardware validation boundary",
            "does **not** certify execution on physical CUDA, ROCm, Apple Metal/MPS, or Intel XPU hardware.",
        )),
        ("native dependency/device readiness regression tests", native_runtime_status_tests, (
            "test_runtime_status_reports_required_and_optional_dependency_readiness",
            "test_runtime_status_fails_closed_for_missing_or_unsupported_dependencies",
            "test_runtime_status_fails_closed_when_no_device_can_initialize",
            "test_runtime_status_recommends_the_first_available_accelerator_then_cpu",
            "test_runtime_status_reports_loaded_architecture_device_and_memory_mode",
        )),
        ("native API status regression", native_engine_api_tests, (
            "test_engine_status_api_exposes_native_dependency_and_device_readiness",
        )),
        ("native architecture regression tests", native_engine_tests, (
            "test_native_config_accepts_llama_mistral_qwen2_and_qwen3",
            "test_dynamic_ntk_rope_scaling_extends_context_and_matches_frequency_formula",
            "test_native_config_supports_yarn_rope_scaling_formula",
            "test_native_grouped_query_attention_matches_expanded_reference",
            "test_native_generate_batch_bounds_same_length_microbatches_and_preserves_order",
        )),
        ("native pull core/API tests", native_model_studio_tests, (
            "test_native_hub_pull_installs_only_supported_safetensors_assets",
            "test_native_hub_pull_rejects_unsupported_architecture_before_weights",
        )),
        ("native pull CLI progress/cancellation tests", native_model_studio_cli_tests, (
            "test_model_studio_native_download_receives_progress_and_cancel_callbacks",
            "test_model_studio_native_download_reports_cooperative_cancel",
        )),
        ("native engine API/CLI pull tests", native_engine_api_tests + native_engine_cli_tests, (
            "test_native_engine_pull_endpoint_starts_model_studio_job",
            "test_engine_pull_downloads_native_model_for_axiom_runtime",
        )),
        ("native Model Studio browser acquisition coverage", native_model_studio_e2e, (
            'config: { operation: "pull_native", repo_id: "Qwen/test" }',
            "progress_bytes: 1024, total_bytes: 4096",
            'getByRole("button", { name: "Cancel operation" })',
        )),
        ("native inference and benchmark browser lifecycle coverage", native_engine_e2e, (
            "local_only: true", "remote_code: false", "engineLoaded = true",
            'getByRole("button", { name: "Run with AXIOM Engine" })',
            'getByRole("button", { name: "Unload model" })',
            'getByLabel("Execution device", { exact: true })',
            'option[value="cuda"]',
            'option[value="rocm"]',
            'option[value="mps"]',
            'option[value="xpu"]',
            'expect(requestBody.device).toBe("cpu")',
            'getByRole("button", { name: "Stop benchmark" })',
            "Benchmark cancelled; AXIOM has been asked to stop the native generation loop.",
            "benchmarkAbortObserved = true",
        )),
        ("native benchmark core metrics and bounded loop", engine_source, (
            "def benchmark_local(",
            '"warmup_runs_requested"',
            '"median_time_to_first_token_ms"',
            '"p95_elapsed_ms"',
            '"generation_tokens_per_second"',
            '"layer_weight_bytes_staged"',
            '"disk_layer_weight_bytes_read"',
            '"kv_cache_bytes_offloaded"',
            "Native benchmark was cancelled.",
        )),
        ("native benchmark API bounds and disconnect cancellation", engine_api, (
            '@router.post("/benchmark")',
            "async def engine_benchmark(",
            "cancel_event.set()",
            "Native benchmark cancelled because the client disconnected.",
        )),
        ("shared API client caller-cancellation propagation", client, (
            "const externalSignal = init.signal;",
            "const abortFromCaller = () => controller.abort(externalSignal?.reason);",
            'externalSignal?.addEventListener("abort", abortFromCaller, { once: true });',
            'externalSignal?.removeEventListener("abort", abortFromCaller);',
            "if (externalSignal?.aborted) throw error;",
        )),
        ("native benchmark CLI surface", native_engine_cli, (
            '@engine_app.command("benchmark")',
            '"--warmup-runs"',
            '"--iterations"',
            '"--max-new-tokens"',
            "benchmark_local(",
        )),
        ("native benchmark GUI telemetry and cancellation", native_engine_panel, (
            "Benchmark native engine",
            "Stop benchmark",
            "Benchmark cancelled; AXIOM has been asked to stop the native generation loop.",
            "median_time_to_first_token_ms",
            "layer_weight_bytes_staged",
            "disk_layer_weight_bytes_read",
            "kv_cache_bytes_offloaded",
        )),
        ("native benchmark core regressions", native_engine_tests, (
            "test_native_benchmark_excludes_warmups_and_never_returns_prompt_or_response",
            "test_native_benchmark_cancellation_stops_before_next_measured_run",
            "test_native_benchmark_rejects_unbounded_work_before_generation",
        )),
        ("native benchmark API regressions", native_engine_api_tests, (
            "test_native_benchmark_api_forwards_bounded_local_options_without_response_payload",
            "test_native_benchmark_api_rejects_iterations_or_output_over_the_limit",
            "test_native_benchmark_api_signals_worker_when_client_disconnects",
        )),
        ("native benchmark CLI regression", native_engine_cli_tests, (
            "test_native_engine_benchmark_cli_forwards_bounded_options_and_prints_no_text",
        )),
        ("native RAM/VRAM estimator formula implementation", engine_source, (
            "def estimate_native_quantization(",
            "embedding_params = vocab * hidden",
            "kv_bytes_total = 2 * layers * kv_heads * head_dim * kv_cache_context * batch_size * kv_bytes",
            "kv_prefill_temporary_bytes = (",
            "total_bytes = unbuffered_total * 1.20",
            '"uncertainty_range_percent": {"low": -15, "high": 25}',
        )),
        ("native estimator exact numerical and Mistral regressions", native_engine_api_tests, (
            "test_quantization_estimator_compares_formats_against_memory_budgets",
            "expected_numeric = {",
            "test_quantization_estimator_separates_mistral_retained_cache_and_prefill_temporary",
            'assert bf16["estimated_total_gib"] == 0.000006706',
        )),
        ("native estimator CLI numerical regressions", native_engine_cli_tests, (
            "test_engine_estimate_cli_compares_formats_against_budgets",
            'assert candidates["bf16"]["estimated_total_gib"] == 0.000003988',
        )),
        ("native estimator calibration boundary documentation", native_engine_docs, (
            "### Memory estimate calibration boundary",
            "not calibrated against physical peak allocations",
            "weight-storage overhead",
            "Mistral prefill temporary memory",
        )),
    )
    for label, source, tokens in native_readiness_contracts:
        for token in tokens:
            if token not in source:
                failures.append(f"R087 native-engine {label} contract is missing: {token}")
    cli_main_source = (root / "axiom/cli/main.py").read_text(encoding="utf-8")
    cli_extended_source = (root / "axiom/cli/extended.py").read_text(encoding="utf-8")
    cli_json_tests = (root / "tests/test_cli_json.py").read_text(encoding="utf-8")
    cli_surface_tests = (root / "tests/test_cli_surface.py").read_text(encoding="utf-8")
    cli_guide = (root / "docs/CLI-GUIDE.md").read_text(encoding="utf-8")
    for token in ('"--verbose"', '"--quiet"', '"--json-errors"', 'def _cli_runtime_options(', 'def _cli_error('):
        if token not in cli_main_source:
            failures.append("Global CLI output controls are missing: " + token)
    for token in (
        "test_global_cli_output_options_are_exposed_before_subcommands",
        "test_quiet_mode_suppresses_normal_command_output",
        "test_json_errors_emit_stable_machine_readable_envelope",
    ):
        if token not in cli_surface_tests:
            failures.append("Global CLI output regression coverage is missing: " + token)
    if "## Global output controls" not in cli_guide:
        failures.append("CLI guide is missing the global output controls reference.")
    workflow_core_source = (root / "axiom/core/workflows.py").read_text(encoding="utf-8")
    workflow_api_source = (root / "axiom/api/workflows.py").read_text(encoding="utf-8")
    workflow_cli_source = (root / "axiom/cli/workflows.py").read_text(encoding="utf-8")
    workflow_client_source = (root / "src/api/client.ts").read_text(encoding="utf-8")
    workflow_ui_source = (root / "src/App.tsx").read_text(encoding="utf-8")
    workflow_e2e_source = (root / "e2e/workflows.spec.ts").read_text(encoding="utf-8")
    workflow_pagination_contracts = (
        (workflow_core_source, ('def list_page(', '"next_cursor":', '"filters": {"search": needle}')),
        (workflow_api_source, ('after_workflow_id:', 'search=q', 'list_page(')),
        (workflow_cli_source, ('"--after"', '"--search"', '"--page-json"')),
        (workflow_client_source, ("workflowsPage:", "after_workflow_id")),
        (workflow_ui_source, ("loadMoreWorkflows", "Load more workflows", "Search saved workflows")),
        (workflow_core_source, ('"next_cursor":', '"has_more":')),
    )
    for source, tokens in workflow_pagination_contracts:
        for token in tokens:
            if token not in source:
                failures.append("Workflow cursor pagination is missing contract token: " + token)
    trace_core_source = (root / "axiom/core/runtime_traces.py").read_text(encoding="utf-8")
    trace_api_source = (root / "axiom/api/diagnostics.py").read_text(encoding="utf-8")
    trace_client_source = (root / "src/api/client.ts").read_text(encoding="utf-8")
    trace_panel_source = (root / "src/components/RuntimeTracePanel.tsx").read_text(encoding="utf-8")
    trace_tests_source = (root / "tests/test_runtime_traces.py").read_text(encoding="utf-8")
    for source, tokens in (
        (trace_core_source, ("def list_runtime_traces(", '"has_more":', '"capacity_reached":', '"filters": {"search": needle}')),
        (trace_api_source, ("offset: int = Query", "search: str = Query", "list_runtime_traces(limit, offset, search)")),
        (trace_client_source, ("runtimeTraces: (limit = 50, offset = 0, search = \"\")", "offset: number;", "has_more: boolean;")),
        (trace_panel_source, ("Load older traces", "Search runtime trace metadata", "loadMore")),
        (trace_tests_source, ("test_trace_listing_paginates_newest_first_and_filters_metadata_only", "test_trace_listing_rejects_invalid_offsets")),
    ):
        for token in tokens:
            if token not in source:
                failures.append("Runtime trace pagination is missing contract token: " + token)
    for token in ('def model_list(', 'def model_add(', 'def model_update(', 'def model_export(', 'def model_import(', 'def model_remove(', 'def model_favorite(', 'def model_info(', 'def model_inspect(', 'def model_analyze(', 'def dataset_inspect(', 'def dataset_clean(', 'def integration_list(', 'def system_info('):
        if token not in cli_main_source:
            failures.append("CLI JSON output support is missing from a primary command: " + token)
    for token in ("def model_search(", "def dataset_validate(", "def dataset_stats(", "def dataset_inspection_payload("):
        if token not in cli_extended_source:
            failures.append("CLI JSON output support is missing from an extended command: " + token)
    for token in (
        '["model", "list", "--json"]',
        '["model", "info", "org/tiny", "--json"]',
        '["model", "inspect", str(model), "--json"]',
        '["dataset", "inspect", str(dataset), "--json"]',
        '["integration", "list", "--json"]',
        '["model", "analyze", "org/tiny", "--json"]',
        '["system", "info", "--json"]',
        '["dataset", "clean", str(dataset), "--output", str(output), "--json"]',
        "test_model_add_and_update_json_preserve_metadata_fields",
        "test_model_update_validation_failure_does_not_mutate_registry",
        '["model", "export", "--path", str(export_path), "--json"]',
        '["model", "import", str(export_path), "--json"]',
        '["model", "remove", first_model["id"], "--yes", "--json"]',
        "test_model_remove_requires_confirmation_and_never_deletes_weight_files",
        '["model", "favorite", first_model["id"], "--unfavorite", "--json"]',
    ):
        if token not in cli_json_tests:
            failures.append("CLI JSON acceptance coverage is missing: " + token)
    engine_cli_source = (root / "axiom/cli/engine.py").read_text(encoding="utf-8")
    engine_cli_tests = (root / "tests/test_axiom_engine_cli.py").read_text(encoding="utf-8")
    for token in ('@engine_app.command("discover")', "def engine_discover(", "discover_local_engine_assets()", "generate_local_stream(", '"--stream"', '"--json"'):
        if token not in engine_cli_source:
            failures.append("Native Engine CLI discovery wiring is missing: " + token)
    for token in ('["discover", "--json"]', "test_engine_discover_json_lists_local_model_and_adapter_metadata", '"--stream"', "test_engine_run_stream_shows_live_output_and_final_metrics"):
        if token not in engine_cli_tests:
            failures.append("Native Engine CLI discovery regression coverage is missing: " + token)
    telemetry_panel = (root / "src/components/EngineTelemetryMonitor.tsx").read_text(encoding="utf-8")
    telemetry_e2e = (root / "e2e/telemetry-monitor.spec.ts").read_text(encoding="utf-8")
    telemetry_history_core = (root / "axiom/core/resource_telemetry.py").read_text(encoding="utf-8")
    telemetry_history_api = (root / "axiom/api/engine.py").read_text(encoding="utf-8")
    telemetry_history_core_tests = (root / "tests/test_resource_telemetry.py").read_text(encoding="utf-8")
    telemetry_history_api_tests = (root / "tests/test_resource_telemetry_api.py").read_text(encoding="utf-8")
    failures.extend(resource_monitoring_issues(
        engine_cli_source, telemetry_panel, app_source, telemetry_e2e,
        telemetry_history_core, telemetry_history_api, client,
        telemetry_history_core_tests, telemetry_history_api_tests,
    ))
    benchmark_trend_panel = (root / "src/components/LocalAxiomEnginePanel.tsx").read_text(encoding="utf-8")
    benchmark_trend_chart = (root / "src/components/BenchmarkResourceTrends.tsx").read_text(encoding="utf-8")
    benchmark_trend_e2e = (root / "e2e/axiom-engine.spec.ts").read_text(encoding="utf-8")
    failures.extend(benchmark_resource_trend_issues(
        benchmark_trend_panel, benchmark_trend_chart, benchmark_trend_e2e,
    ))
    telemetry_export_monitor = telemetry_panel
    telemetry_export_e2e = (root / "e2e/telemetry-monitor.spec.ts").read_text(encoding="utf-8")
    telemetry_export_docs = (root / "docs/API.md").read_text(encoding="utf-8")
    failures.extend(telemetry_export_privacy_issues(
        telemetry_export_monitor, telemetry_export_e2e, telemetry_export_docs,
    ))
    package_verifier = (root / "scripts/verify_package_artifacts.py").read_text(encoding="utf-8")
    package_tests = (root / "tests/test_package_artifact_verification.py").read_text(encoding="utf-8")
    package_docs = (root / "docs/Packaging.md").read_text(encoding="utf-8")
    package_workflow = (root / ".github/workflows/quality-gates.yml").read_text(encoding="utf-8")
    failures.extend(package_verification_issues(
        package_verifier, package_workflow, package_tests, package_docs,
    ))
    schema_docs = (root / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    schema_sources = {
        "workflows": (root / "axiom/core/workflows.py").read_text(encoding="utf-8"),
        "llm_profiles": (root / "axiom/core/llm_profiles.py").read_text(encoding="utf-8"),
        "security_events": (root / "axiom/core/security_events.py").read_text(encoding="utf-8"),
        "hardware_snapshots": (root / "axiom/core/hardware_snapshots.py").read_text(encoding="utf-8"),
        "resource_telemetry": (root / "axiom/core/resource_telemetry.py").read_text(encoding="utf-8"),
        "workspaces": (root / "axiom/core/workspaces.py").read_text(encoding="utf-8"),
        "prediction_jobs": (root / "axiom/core/prediction_jobs.py").read_text(encoding="utf-8"),
        "integrations": (root / "axiom/core/integrations.py").read_text(encoding="utf-8"),
        "permissions": (root / "axiom/core/tool_permissions.py").read_text(encoding="utf-8"),
        "registry": (root / "axiom/models/registry.py").read_text(encoding="utf-8"),
        "workspace_metadata": (root / "axiom/core/workspace_metadata.py").read_text(encoding="utf-8"),
        "fine_tuning": (root / "axiom/core/fine_tuning.py").read_text(encoding="utf-8"),
        "model_studio": (root / "axiom/core/model_studio.py").read_text(encoding="utf-8"),
        "runtime_traces": (root / "axiom/core/runtime_traces.py").read_text(encoding="utf-8"),
    }
    failures.extend(persisted_schema_inventory_issues(schema_docs, schema_sources))
    for path in (
        "/api/health", "/api/meta", "/api/workspace/runtime",
        "/api/workspace/models?offset=0&limit=5", "/api/workspace/datasets",
        "/api/workspace/events", "/api/diagnostics/workspace",
        "/api/diagnostics/roadmap", "/api/diagnostics/runtime-connectors",
    ):
        if path not in client:
            failures.append(f"API playground allowlist is missing {path}.")
    failures.extend(api_playground_issues(component, client))
    contract_examples_core = (root / "axiom/core/api_examples.py").read_text(encoding="utf-8")
    contract_examples_api = (root / "axiom/api/examples.py").read_text(encoding="utf-8")
    contract_examples_cli = (root / "axiom/cli/api.py").read_text(encoding="utf-8")
    contract_examples_component = (root / "src/components/ApiPlayground.tsx").read_text(encoding="utf-8")
    contract_examples_e2e = (root / "e2e/api-playground.spec.ts").read_text(encoding="utf-8")
    contract_examples_tests = (root / "tests/test_api_examples.py").read_text(encoding="utf-8")
    contract_examples_docs = (root / "docs/API.md").read_text(encoding="utf-8")
    contract_examples_server = (root / "axiom/api/server.py").read_text(encoding="utf-8")
    failures.extend(api_contract_examples_issues(
        contract_examples_core, contract_examples_api, contract_examples_server,
        contract_examples_cli, cli_main_source, client, contract_examples_component,
        contract_examples_e2e, contract_examples_tests, contract_examples_docs,
    ))
    snapshot_core = (root / "axiom/core/hardware_snapshots.py").read_text(encoding="utf-8")
    snapshot_api = (root / "axiom/api/hardware_snapshots.py").read_text(encoding="utf-8")
    snapshot_panel = (root / "src/components/HardwareSnapshotPanel.tsx").read_text(encoding="utf-8")
    snapshot_e2e = (root / "e2e/hardware-snapshots.spec.ts").read_text(encoding="utf-8")
    snapshot_tests = (
        (root / "tests/test_hardware_snapshots.py").read_text(encoding="utf-8")
        + (root / "tests/test_hardware_snapshots_api.py").read_text(encoding="utf-8")
        + (root / "tests/test_hardware_snapshots_cli.py").read_text(encoding="utf-8")
    )
    snapshot_docs = (
        (root / "docs/API.md").read_text(encoding="utf-8")
        + (root / "docs/wiki/CLI-Reference.md").read_text(encoding="utf-8")
        + (root / "docs/Storage-Schemas-and-Migrations.md").read_text(encoding="utf-8")
    )
    failures.extend(hardware_snapshot_issues(
        snapshot_core, snapshot_api, (root / "axiom/api/server.py").read_text(encoding="utf-8"),
        cli_main_source, client, snapshot_panel, app_source, snapshot_e2e,
        snapshot_tests, snapshot_docs,
    ))
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

    fabric_profiles_core = (root / "axiom/core/engine_fabric_profiles.py").read_text(encoding="utf-8")
    fabric_profiles_api = (root / "axiom/api/engine.py").read_text(encoding="utf-8")
    fabric_profiles_cli = (root / "axiom/cli/engine.py").read_text(encoding="utf-8")
    fabric_profiles_ui = (root / "src/components/LocalEngineFabricPanel.tsx").read_text(encoding="utf-8")
    fabric_profiles_client = (root / "src/api/client.ts").read_text(encoding="utf-8")
    fabric_profiles_e2e = (root / "e2e/engine-fabric.spec.ts").read_text(encoding="utf-8")
    fabric_profiles_docs = (root / "docs/Engine-Fabric.md").read_text(encoding="utf-8")
    if not all(token in fabric_profiles_core for token in ("MAX_PROFILES = 25", "custom_endpoints_persisted", "credentials_persisted")):
        failures.append("Engine Fabric workspace profile storage is missing its bounds or secret-exclusion contract.")
    if not all(token in fabric_profiles_api for token in ('@router.get("/backends/profiles")', '@router.post("/backends/profiles", status_code=201)', '@router.delete("/backends/profiles/{profile_id}")', '@router.post("/backends/profiles/{profile_id}/connect")')):
        failures.append("Engine Fabric profile CRUD and one-shot verification routes are incomplete.")
    if not all(token in fabric_profiles_cli for token in ('@engine_profile_app.command("save")', '@engine_profile_app.command("list")', '@engine_profile_app.command("connect")', '@engine_profile_app.command("delete")')):
        failures.append("Engine Fabric CLI profile commands are incomplete.")
    if not all(token in fabric_profiles_client for token in ("engineBackendProfiles", "saveEngineBackendProfile", "deleteEngineBackendProfile", "connectEngineBackendProfile")):
        failures.append("Engine Fabric profile API methods are incomplete in the typed client.")
    if not all(token in fabric_profiles_ui for token in ("Saved local connector profiles", "Connect", "Disconnect", "API keys", "endpoint URLs")):
        failures.append("Engine Fabric GUI profile lifecycle controls are incomplete.")
    if not all(token in fabric_profiles_e2e for token in ("Saved Engine Fabric profile", "VERIFIED THIS SESSION", "NOT CONNECTED", "external runtime remains running")):
        failures.append("Engine Fabric connector profile lifecycle is missing browser acceptance coverage.")
    if "one-shot" not in fabric_profiles_docs or "does not stop" not in fabric_profiles_docs:
        failures.append("Engine Fabric profile documentation must describe stateless connection and disconnect semantics.")

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
        "e2e/runtime-connectors.spec.ts", "e2e/agent-profiles.spec.ts", "e2e/engine-fabric.spec.ts",
        "e2e/model-registry-filters.spec.ts", "e2e/inference-benchmark.spec.ts",
        "e2e/runtime-traces.spec.ts", "e2e/telemetry-monitor.spec.ts", "e2e/hardware-snapshots.spec.ts", "e2e/ollama-service.spec.ts", "e2e/settings-retention.spec.ts", "e2e/datasets.spec.ts", "e2e/embeddings.spec.ts", "e2e/reranking.spec.ts", "e2e/knowledge.spec.ts", "e2e/security-supply-chain.spec.ts",
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
