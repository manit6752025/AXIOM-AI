# AXIOM Roadmap

<!-- AXIOM-ROADMAP-UPDATE:START -->
## Product roadmap — 120 scoped deliverables

This backlog expands the ten original priorities into concrete, reviewable
deliverables. Preserve the existing Control Center navigation, visual language,
responsive layout, and keyboard behavior while extending its workflows. Items
are planned unless explicitly marked otherwise; scope is not a claim that the
feature exists or a delivery-date promise.

### 01 · Platform foundation

- [R001 · Implemented; tests passing] Publish a versioned API contract and compatibility policy.
- [R002 · Implemented; tests passing] Resolve the workspace using explicit override, `AXIOM_WORKSPACE`, nearest parent `axiom.yaml`, then the current directory; reject explicit symlink roots.
- [R003 · In progress; validation pending] Make API configuration environment-backed and documented; GUI port now accepts AXIOM_GUI_PORT, CORS origins and trusted model roots are documented, and a cross-platform compatibility report is being added.
- [R004 · Implemented; tests passing] Normalize validation and error response envelopes.
- [R005 · Implemented; tests passing] Add request IDs and correlation across CLI, API, and GUI.
- [R006 · In progress; validation pending] Add bounded pagination to every collection endpoint.
- [R007 · In progress; validation pending] Define stable resource IDs and schema migration behavior.
- [R008 · In progress; validation pending] Add cancellation and timeout semantics to long operations; AXIOM-native streaming now supports cooperative cancellation, stop-string termination, and explicit finish reasons, while other long operations still need cancellation support.
- [R009 · Planned] Separate read-only plans from side-effecting jobs in API scopes.
- [R010 · Implemented; tests passing] Add read-only workspace health checks for config parsing, path safety, file permissions, free disk, Python compatibility, and required dependency imports.
- [R011 · In progress; validation pending] Document data retention, browser-local account settings, in-memory request traces, workspace metadata, generated files, and deletion boundaries in Settings and the retention guide.
- [R012 · In progress; validation pending] Add automated Python/Node/OS runtime compatibility checks to cross-platform release CI.

### 02 · CLI experience

- [R013 · Implemented; tests passing] Add discoverable command groups for workflow and evaluation.
- [R014 · In progress; validation pending] Add consistent JSON output to every list and inspect command; model add/list/info/inspect/analyze/search/import/export/remove/favorite, integration list, system info, and dataset inspect/validate/stats/clean now have versioned outputs and focused CLI tests. Remaining command families still need coverage.
- [R015 · In progress; validation pending] Expose and document Typer shell completion for Bash, Zsh, Fish, and PowerShell; add no-write completion smoke coverage.
- [R016 · Planned] Add interactive prompts only where a non-interactive flag exists.
- [R017 · Planned] Add global verbosity, quiet, and machine-readable error modes.
- [R018 · Planned] Add operation progress and cancellation for downloads and scans.
- [R019 · Implemented; tests passing] Add workspace-aware `axiom status` summaries with JSON health, workspace path, version, and Git working-tree state.
- [R020 · Planned] Add import/export commands for workspace metadata.
- [R021 · Planned] Add plan diff and reproducible plan printing.
- [R022 · In progress; validation pending] Add configuration validation and safe repair suggestions; validation reports stable error codes and non-mutating suggestions for missing, malformed, oversized, symlinked, or wrong-shaped YAML.
- [R023 · Planned] Add clear exit-code documentation and stable failure codes.
- [R024 · Planned] Keep CLI/API feature parity documented in the command reference.

### 03 · Control Center GUI

- [R025 · Planned] Preserve current sidebar, topbar, theme, and page transitions.
- [R026 · Planned] Add shared accessible form, table, dialog, and empty-state patterns.
- [R027 · In progress; validation pending] Add a global command palette with page and action search.
- [R028 · In progress; validation pending] Add deep links and browser-history navigation for workspace pages.
- [R029 · Planned] Add global loading, retry, stale-data, and offline indicators.
- [R030 · In progress; validation pending] Add server-side search, sorting, and saved filter presets alongside bounded pagination for large inventories.
- [R031 · In progress; validation pending] Add consistent confirmation for destructive and irreversible actions.
- [R032 · Planned] Add keyboard navigation and screen-reader status announcements.
- [R033 · Planned] Add responsive compact layouts for narrow and touch screens.
- [R034 · Planned] Add user-controlled refresh cadence and reduced-motion handling.
- [R035 · In progress; validation pending] Add export/download affordances with clear local-file boundaries.
- [R036 · Planned] Add frontend error reporting that redacts secrets and data values.

### 04 · Model lifecycle and compatibility

- [R037 · In progress; validation pending] Expand registry metadata with architecture, size, source, and license; CLI add/update, schema-checked import/export, and hash/tag fields are exposed, while registry/UI validation remains to be completed.
- [R038 · Implemented; tests passing] Add safe local-folder discovery without following directory symlinks; file symlinks are accepted only when resolved targets remain inside trusted roots.
- [R039 · In progress; validation pending] Add model inspection summaries in the GUI.
- [R040 · In progress; validation pending] Add duplicate detection by source and content metadata.
- [R041 · In progress; validation pending] Add model tags, favorites, and workspace labels.
- [R042 · Planned] Add format/runtime compatibility matrices with version evidence.
- [R043 · In progress; validation pending] Add metadata-based RAM/VRAM fit ranges with explicit uncertainty bands and conservative memory reserves in the model registry.
- [R044 · In progress; validation pending] Surface format and hardware-fit warnings in the existing Models page; runtime loadability and non-NVIDIA GPU measurement remain unverified.
- [R045 · Planned] Add model source attribution and license review state.
- [R046 · Planned] Add optional Hugging Face metadata lookup with consent and timeout.
- [R047 · In progress; validation pending] Add safe model removal that never deletes weights implicitly.
- [R048 · In progress; validation pending] Add registry import/export with schema validation and migration.

### 05 · Dataset quality and privacy

- [R049 · Implemented; tests passing] Add selectable bounded scan budgets and visible truncation causes.
- [R050 · Implemented; tests passing] Add schema comparison across train, validation, and test datasets.
- [R051 · Implemented; tests passing] Add configurable required-field and type checks.
- [R052 · Implemented; tests passing] Report UTF-8 errors, newline styles, malformed/non-object records, and JSONL delimiter applicability.
- [R053 · Implemented; tests passing] Configure duplicate-key fields and alert thresholds.
- [R054 · Implemented; tests passing] Show aggregate whitespace-token length bins without returning text values.
- [R055 · Implemented; tests passing] Configure sensitive-name substring rules and exact-name false-positive exclusions.
- [R056 · Implemented; tests passing] Add bounded, explicit-opt-in local PII heuristics that return redacted counts only.
- [R057 · Implemented; tests passing] Add optional parent-manifest SHA-256 lineage links to bounded provenance manifests.
- [R058 · Implemented; tests passing] Add no-write cleaning previews with explainable row-change counts.
- [R059 · Implemented; tests passing] Add exportable validation reports with scan-limit metadata.
- [R060 · Implemented; tests passing] Add explicit deletion controls for in-memory dataset reports; no disk cache is retained.

### 06 · Training plans and visual workflows

- [R061 · In progress; validation pending] Add a visual workflow graph using the existing GUI design system.
- [R062 · In progress; validation pending] Define a versioned workflow schema for nodes, ports, and edges.
- [R063 · In progress; validation pending] Add validated dataset, model, plan, evaluation, and export nodes.
- [R064 · In progress; validation pending] Detect missing inputs, invalid edges, and dependency cycles.
- [R065 · In progress; validation pending] Add workflow templates for common supervised fine-tuning paths.
- [R066 · In progress; validation pending] Add autosave, rename, duplicate, and version history for workflows.
- [R067 · In progress; validation pending] Add backend-owned workflow validation before execution is possible.
- [R068 · Implemented; tests passing] Add planning-only dry-run summaries with estimated resource requirements.
- [R069 · Planned] Add explicit job execution states and cancellation contracts.
- [R070 · Planned] Add checkpoint, resume, and output-path policy controls.
- [R071 · In progress] Export seed and software-version manifests; seeds remain unapplied until an execution runner exists.
- [R072 · Implemented; tests passing] Keep planning-only states visually distinct from executed training.

### 07 · Experiment tracking and evaluation

- [R073 · Implemented; tests passing] Add editable experiment metadata and user-defined tags without mutating saved plans.
- [R074 · Implemented; tests passing] Add immutable offline evaluation records with input and output manifests.
- [R075 · Implemented; tests passing] Persist exact plan snapshots across API, GUI, and CLI; reject stale GUI snapshots.
- [R076 · Implemented; tests passing] Add experiment search, exact-tag filters, pagination, and deletion controls.
- [R077 · Implemented; tests passing] Add evaluation dataset selection and split provenance.
- [R078 · Planned] Add benchmark adapters with explicit model/runtime requirements.
- [R079 · Implemented; tests passing] Add documented exact-match and micro token-overlap metrics for precomputed text outputs.
- [R080 · Implemented; tests passing] Add immutable baseline creation, selection, and comparison.
- [R081 · Implemented; tests passing] Add configurable score-drop thresholds and regression gates.
- [R082 · Implemented; tests passing] Add side-by-side report scores, Wilson confidence intervals, and sample counts.
- [R083 · Implemented; tests passing] Add exportable evaluation reports with input/output hashes and software metadata.
- [R084 · Implemented; tests passing] Add explicit deletion controls for saved plan and evaluation records.

### 08 · Runtime, optimization, and quantization

- [R085 · Implemented; tests passing] Add passive runtime connector discovery and capability reporting without assuming a connection or credentials.
- [R086 · In progress; validation pending] Add explicit connect/disconnect controls with secret-safe storage. The unified LLM client has explicit cloud consent and request-only key handling; AXIOM now has explicit start/stop controls for a verified AXIOM-managed Ollama service plus bounded, cancellable model-pull jobs. Persistent provider profiles and broader connector lifecycle controls remain outstanding.
- [R087 · In progress; validation pending] Report native-engine dependency readiness, supported Llama/Mistral/Qwen2/Qwen3 floating-point safetensors checkpoints, public native-model acquisition with staged installs/progress, explicit CPU/CUDA/ROCm/Metal/XPU selection, loaded backend/architecture, and local-only execution scope. Native decoding now includes validated greedy/temperature sampling, exact-count top-k filtering, nucleus top-p, probability-relative min-p, repetition penalty, reproducible sampling seeds, and stop-sequence handling, wired through the native GUI, CLI, synchronous/streaming API, unified local LLM client, and compatible Engine Fabric routes. Experimental in-RAM layer-by-layer staging and disk-resident safetensors layer streaming with CPU KV-cache offload are available with transfer telemetry. Mistral sliding-window attention now retains a compact bounded KV suffix per layer while preserving absolute RoPE positions and context limits. Grouped-query attention passes compact KV heads directly to SDPA where `enable_gqa` is supported, with expanded-SDPA and manual-math compatibility fallbacks; regression coverage compares both paths with an expanded reference. Native forward calls now validate KV-cache layer count, K/V tensor shapes, head dimensions, model dtype, memory-mode device, positive and per-layer sequence lengths before tensor operations, with malformed-cache regression coverage. Native config loading rejects boolean/coercible dimensions, model_type/architecture mismatches, malformed BOS/EOS IDs, and invalid tied-embedding flags. RoPE now supports default, linear interpolation, and Llama 3 wavelength-based scaling; dynamic NTK, YaRN, and other scaling modes remain unsupported until implemented and reference-tested. Runtime forward inputs now reject non-tensors, empty sequences, unsupported index dtypes, and prompt token IDs outside the checkpoint vocabulary before model execution. Rotary position embeddings are now computed once per forward sequence and shared by query/key projections across layers, avoiding repeated frequency-table work; checkpoint loading rejects non-floating embedding weights before selecting auto precision. Native checkpoint loading now also validates the vector dimensions of optional q/k/v/o and MLP projection biases plus an optional output-head bias before runtime execution; Qwen2 bias loading has a direct forward-pass regression test. Ordinary non-sliding prefill now delegates causal masking to SDPA without constructing a dense quadratic boolean mask, and single-token decode over a populated cache uses the all-available-key fast path; explicit masks remain for chunked and sliding-window attention. Cache and GQA changes still require full CI, release, and physical-device validation. Engine Fabric probes fixed local Ollama, llama.cpp, LM Studio, vLLM and llama-cpp-python APIs, documents their trade-offs, recommends workload profiles, and offers bounded loopback chat routing; min-p is routed only to native/Ollama adapters rather than silently ignored by other backends. It does not convert external runtime formats into native weights. AXIOM-native GGUF ingestion supports dense Llama/Mistral/Qwen2/Qwen3 checkpoints in single-file or metadata-validated split-shard form, using gguf-py dequantization and layer-on-demand loading in streaming modes. Execution remains floating-point rather than low-bit kernels, and tokenizer fidelity is limited to a sidecar/tokenizer JSON or supported embedded tokenizer types. EXL2/GPTQ/AWQ native kernels and broader architectures remain outstanding.
- [R088 · In progress; validation pending] Report actual local inference wall time, prefill latency, cached-decode latency, time to first token, generated tokens/second, and layer-streaming transfer counters from AXIOM's native token loop; wall time includes stream processing while throughput is based on prefill+decode compute time. Peak process/device memory telemetry and physical-hardware validation remain outstanding.
- [R089 · Implemented; tests passing] Add prompt/token-limit inputs, excluded warm-up runs, bounded measured iterations, and explicit concurrency controls to the existing Runtime page.
- [R090 · In progress; validation pending] Provide parameter-based memory estimates with documented weight/KV-cache/workspace formulas, overheads, safety margin, and uncertainty range. Core, API, and Control Center paths exist; `axiom engine estimate` now exposes the same estimate without loading weights. These are estimates, not physical-memory measurements.
- [R091 · In progress; validation pending] Compare candidate memory estimates against supplied RAM and VRAM budgets and recommend only among currently native-supported precisions. INT4/INT8 remain sizing-only and are explicitly not loadable by AXIOM Native Engine; budget fit does not guarantee load success.
- [R092 · Planned] Keep estimated fit separate from measured quality and speed.
- [R093 · Implemented; tests passing] Add bounded in-memory runtime request traces with request IDs, metadata-only records, export, and explicit clear controls.
- [R094 · Implemented; tests passing] Add explicit model unload and cache cleanup, plus one-model-at-a-time cache replacement; framework allocator release remains best-effort.
- [R095 · In progress; validation pending] Add adapter compatibility checks for CPU, CUDA, ROCm, Metal/MPS, and Intel XPU; metadata/tensor-shape preflight and per-device availability reporting are wired, but real execution on each device and the release test gate remain to be validated.
- [R096 · In progress; validation pending] Retry CPU after recognized accelerator out-of-memory failures in auto mode and report the selected backend/reason; broader device-driver recovery remains to be validated.

### 09 · Agent profiles, MCP, and security

- [R097 · Planned] Add versioned agent profiles with explicit task capabilities.
- [R098 · In progress; validation pending] Enforce default-deny tool permissions in the backend boundary.
- [R099 · Planned] Add per-tool read, write, network, and execution scopes.
- [R100 · Planned] Add auditable allow/deny decisions with redacted arguments.
- [R101 · Planned] Add MCP server inventory and protocol handshake inspection.
- [R102 · Planned] Add tool/resource schema browsing without tool execution by default.
- [R103 · Planned] Add per-server trust, ownership, and connection state.
- [R104 · Planned] Add safe secret references and never render stored secret values.
- [R105 · Planned] Add config, permissions, and exposed-network security audits.
- [R106 · Planned] Add path traversal and symlink threat-model regression coverage.
- [R107 · Planned] Add dependency, package, and release-artifact security checks.
- [R108 · Planned] Add security event export and administrator review workflows.

### 10 · API, telemetry, diagnostics, and delivery

- [R109 · Implemented; tests passing] Add an allowlisted API playground with bounded payloads and timeouts.
- [R110 · Planned] Add typed API examples generated from the supported contract.
- [R111 · Planned] Add local hardware snapshot history with opt-in retention.
- [R112 · In progress; validation pending] Expose point-in-time CPU, process RSS, system RAM, device allocator, and trusted-model-storage telemetry via `GET /api/engine/telemetry` and `axiom engine telemetry --json`; the endpoint omits raw paths and user content. Historical sampling/export and physical-hardware validation remain outstanding.
- [R113 · Planned] Add benchmark-linked resource trend charts with missing-data labels.
- [R114 · Planned] Add telemetry export, clear-history, and privacy controls.
- [R115 · Planned] Add diagnostic checks for config, paths, dependencies, and storage.
- [R116 · Planned] Add copyable support bundles with explicit secret/data redaction.
- [R117 · Implemented; tests passing] Add release smoke checks for CLI, API, production UI build, and browser acceptance coverage together.
- [R118 · Planned] Add cross-platform packaging verification for supported targets.
- [R119 · Planned] Add migration/rollback notes for every persisted-data schema change.
- [R120 · Planned] Add end-to-end acceptance coverage for each shipped workflow.

### Delivery gates

An item is shipped only when its backend validation and security boundaries,
GUI states, CLI surface where appropriate, tests, and user documentation are
verified. Items marked in progress are not complete until those gates pass.
<!-- AXIOM-ROADMAP-UPDATE:END -->

## Beta.5 — released

- [x] Agent/model optimizer
- [x] Quantization lab
- [x] Hardware-aware optimization planning
- [x] Local daemon health and capability endpoints
- [x] Dataset quality inspection
- [x] Trusted model filesystem boundaries
- [x] Secure administrator storage
- [x] E2E authentication validation

### Next

- Connect the local daemon to model, dataset, training, runtime, and MCP workflows
- Deeper runtime benchmarking
- Broader local-model/runtime integration
- Profiling and telemetry improvements

<!-- AXIOM-BETA5-RELEASE -->

## Beta.5 completed

- Agent/model optimization planning
- Quantization workflow and hardware-aware recommendations
- Security hardening for model filesystem access
- Secure administrator verification storage
- CI browser/E2E reliability improvements for the landing page

## Next

- Expand measured runtime benchmarking across supported backends
- Broaden model format and accelerator coverage
- Continue security/code-quality hardening
- Improve optimizer telemetry and diagnostics

## v0.2.0-beta.5

AXIOM beta.5 adds the optimizer/quantization workflow, hardware-aware model planning, hardened model filesystem boundaries, safer administrator verifier storage, and improved CI/E2E validation. See [the beta.5 release notes](docs/releases/v0.2.0-beta.5.md).

### Beta.5 — released

- Optimizer and quantization lab workflow
- Agent-oriented model planning
- Filesystem/path security hardening
- Administrator storage hardening
- CI/E2E reliability improvements

## v0.2.0-beta.7 — released

- Optimizer and quantization planning improvements.
- Model-path and administrator-storage hardening.
- E2E and release-pipeline verification.

## AXIOM v0.2.0-beta.5 macOS DMG

The release workflow packages separate macOS Apple Silicon (arm64) and Intel (x86_64) DMGs. Each contains the AXIOM CLI executable and compiled Control Center frontend, plus a SHA-256 checksum. The DMGs are currently unsigned and not notarized; API-backed UI features require a separately running/configured AXIOM backend. See `docs/releases/v0.2.0-beta.5.md` for release details.

## v0.2.0-beta.9 — released

- [x] Unified Optimizer and Quantization Lab theme
- [x] Quantization Lab size comparison
- [x] Read-only workspace API on the local daemon
- [x] Web UI served by the daemon at `/ui/`
- [x] Web UI bundled into every release package and the PyPI wheel
- [x] Automatic PyPI publishing on version tags

### Remaining follow-on work

- Finish connecting intentionally disconnected GUI views through validated APIs.
- Add measured runtime benchmarking across supported backends.
- Broaden model format and accelerator coverage.

## Community adoption and validation workstream

The companion [Community and Adoption Strategy](docs/Community-and-Adoption-Strategy.md) proposes a workstream for public benchmark evidence, hardware-estimate calibration, integration quality, physical-hardware validation, user support, and external contribution.

This is a proposal, not a delivery-status claim. It supplements the 120 scoped deliverables above without changing their IDs or marking unverified work as complete. Hardware validation must distinguish simulated CI coverage from measurements on physical NVIDIA, Apple Silicon, AMD, and CPU-only systems where those test environments are available. External repository metrics quoted in assessments must be checked against a current public snapshot before publication.

