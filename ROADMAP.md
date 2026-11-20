# AXIOM Roadmap

<!-- AXIOM-ROADMAP-UPDATE:START -->
## Product roadmap — 140 scoped deliverables

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
- [R006 · Implemented; tests passing] Add bounded pagination to every collection endpoint. Fine-tuning jobs and Agent Profiles now expose bounded API pages; Model Studio job history is exposed in the GUI and CLI with bounded search/pagination; Training history supports inspect/load-more and CLI offsets. Existing Models, Experiments, Datasets, Workflows, Logs, and Runtime Trace pages remain bounded. GitHub Actions verified Python, lint, production build, browser E2E, and Ubuntu/macOS/Windows static contracts.
- [R007 · Implemented; tests passing] Define stable resource IDs and schema migration behavior. New persisted resources use canonical UUID4-hex IDs; legacy model IDs remain deterministic; workflow/experiment/evaluation legacy defaults migrate on next write; unsupported schemas fail closed. Storage schema rules and regression coverage are documented.
- [R008 · Implemented; tests passing] Add cancellation and timeout semantics to long operations. Native streaming, prediction jobs, fine-tuning, Ollama pulls, and Model Studio pull/install/convert workers now support cooperative cancellation and bounded timeouts; Model Studio exposes API and GUI cancellation with distinct cancelled/timed-out states and safe cleanup. Full Python, lint, production build, browser E2E, and Ubuntu/macOS/Windows release contracts passed.
- [R009 · Implemented; tests passing] Separate read-only plans from side-effecting jobs in API scopes. Opt-in read/write bearer tokens protect `/api/*` and `/v1/*`; unclassified mutations require write scope, with documented validation/planning allowlists and 401/403 request-ID envelopes.
- [R010 · Implemented; tests passing] Add read-only workspace health checks for config parsing, path safety, file permissions, free disk, Python compatibility, and required dependency imports.
- [R011 · In progress; validation pending] Document data retention, browser-local account settings, in-memory request traces, workspace metadata, generated files, and deletion boundaries in Settings and the retention guide.
- [R012 · In progress; validation pending] Add automated Python/Node/OS runtime compatibility checks to cross-platform release CI.

### 02 · CLI experience

- [R013 · Implemented; tests passing] Add discoverable command groups for workflow and evaluation.
- [R014 · In progress; validation pending] Add consistent JSON output to every list and inspect command; model add/list/info/inspect/analyze/search/import/export/remove/favorite, integration list, system info, and dataset inspect/validate/stats/clean now have versioned outputs and focused CLI tests. Remaining command families still need coverage.
- [R015 · Implemented; tests passing] Expose and document Typer shell completion for Bash, Zsh, Fish, and PowerShell. Built-in install/show options remain available; `axiom completion <shell>` previews generated scripts without modifying shell configuration, with explicit no-write tests for all supported shells.
- [R016 · Implemented; tests passing] Add interactive prompts only where a non-interactive flag exists. Root `--non-interactive`/`--no-input` mode, TTY-aware prompt guards, environment-backed cloud credentials, explicit remote-send consent, and safe default-deny destructive confirmations are covered by CLI regressions and full cross-platform CI.
- [R017 · Implemented; tests passing] Add global verbosity, quiet, and machine-readable error modes. Shared schema-versioned error reporting now covers root commands and command groups, preserves legacy success JSON and exit codes, and retains human output compatibility. Verified with 724 Python tests, lint, production build, browser E2E, and Ubuntu/macOS/Windows release checks.
- [R018 · Implemented; tests passing] Add operation progress and cancellation for downloads and scans. Bounded dataset profiling, cleaning, preview, manifest hashing, Ollama pulls, native checkpoint downloads, GGUF downloads/installs, and conversions now report progress on interactive terminals and support cooperative Ctrl+C cancellation; atomic cleaner publication and partial download cleanup are covered by tests. Verified with 732 Python tests, lint, production build, browser E2E, and Ubuntu/macOS/Windows release contracts.
- [R019 · Implemented; tests passing] Add workspace-aware `axiom status` summaries with JSON health, workspace path, version, and Git working-tree state.
- [R020 · Implemented; tests passing] Add import/export commands for workspace metadata. Added a versioned, checksummed ZIP archive for the model registry, workflows, experiments, evaluation reports, and agent task profiles; CLI `axiom workspace metadata export/import`, workspace API endpoints, and Settings UI download/upload. Imports validate all members/schemas/limits before a merge or explicit collection replacement, reject conflicts before writes, and stage restrictive-permission files with rollback. Excludes project config, credentials, MCP grants, integrations, job history, datasets, weights, and generated artifacts. Verified with 741 Python tests, ESLint, TypeScript/production build, 35 browser E2E tests, and Ubuntu/macOS/Windows release contracts.
- [R021 · Implemented; tests passing] Add reproducible saved-plan printing and field-level diffs. Canonical JSON snapshots include normalized planning inputs, captured hardware, exact estimates, planner/AXIOM provenance, an explicit non-execution scope marker, and stable SHA-256 fingerprints. Added `axiom experiment print/diff`, read-only API endpoints, and Training-page saved-plan tools; legacy records retain unknown provenance rather than inheriting current versions. Diffs compare inputs, hardware, planner provenance, and estimate fields without rerunning the planner or starting jobs. Verified by focused Python core/CLI/API tests, browser acceptance for GUI print/diff, ESLint, TypeScript/production build, and Ubuntu/macOS/Windows quality gates.
- [R022 · Implemented; tests passing] Add configuration validation and safe repair suggestions. `axiom config validate --path … --json` reports stable codes for missing, symlinked, non-regular, unreadable, oversized, malformed, non-mapping, and wrong-shaped configuration. Reads are bounded to 256 KiB plus one byte and do not follow the final symlink; suggestions are advisory and never mutate the file. Focused CLI tests cover each required class and valid-file no-write behavior. Verified with Python tests, ESLint, production build, browser E2E, and Ubuntu/macOS/Windows static checks; configuration validation remains CLI-local rather than exposing arbitrary filesystem paths over the API.
- [R023 · Implemented; tests passing] Document CLI success (0), operation/validation failure (1), usage/safety rejection (2), and cooperative cancellation (130). Centralize the numeric constants and default failure status; keep stable JSON error codes separate from explanatory messages. Regression coverage verifies command-group error envelopes, Click parse failures, unsafe support-bundle exit status, human-readable mode, and Ctrl+C cancellation. Verified with full Python tests, ESLint, production build, browser E2E, and Ubuntu/macOS/Windows static checks.
- [R024 · Implemented; tests passing] Expand the CLI reference with every registered root command, feature-family mapping to representative typed API endpoints and Control Center pages, and explicit CLI-only host/process commands. Add contract tests to detect drift against the live Typer command tree, `src/api/client.ts`, and `src/App.tsx` navigation. Verified with full Python tests, ESLint, production build, browser E2E, and Ubuntu/macOS/Windows static checks.

### 03 · Control Center GUI

- [R025 · Implemented; tests passing] Preserve the existing responsive sidebar and collapsed state, persistent dark/light theme, topbar controls, active-page navigation, and browser back/forward restoration. Added browser acceptance coverage for desktop collapse/expand, theme persistence across reload, responsive mobile navigation, and history-driven page state. Verified with full browser E2E plus Python, lint, production build, and Ubuntu/macOS/Windows release checks.
- [R026 · Implemented; tests passing] Add shared accessible form, table, dialog, and empty-state patterns. New `FormField`, `DataTable`, `AccessibleDialog`, and `EmptyState` primitives provide associated labels/hints/errors, captioned semantic tables with scoped headers, modal focus management/Escape handling, and polite live announcements. Migrated password-change and saved-plan diff flows, added focused Playwright checks for focus wrapping, descriptions, empty states, and table semantics, and documented contributor use/limitations. Verified with 40 browser tests, Python suite, ESLint, strict TypeScript/production build, and Ubuntu/macOS/Windows static release checks.
- [R027 · Implemented; tests passing] Extend the global Ctrl/Command+K palette to search workspace pages and execute local UI actions, including theme switching, administrator password change, and sign-out. Supports keyboard and pointer selection, arrow-key result navigation, Enter execution, Escape dismissal, active-page indication, and an explicit no-match state. The palette does not run model/workflow jobs or expose destructive operations. Verified with browser acceptance for theme action, password dialog action, page navigation, and empty results; full Python tests, ESLint, production build, browser E2E, and Ubuntu/macOS/Windows static contracts passed.
- [R028 · Implemented; tests passing] Verify direct workspace deep links through `?page=` URLs, URL updates from sidebar navigation, reload persistence, and browser Back/Forward restoration of both the selected page and navigation state. Documented supported links, the API Playground page, and safe fallback for unknown slugs. Browser acceptance covers direct Settings links, reload, Models/Datasets history, `api-playground`, and invalid-page fallback. Verified with Python tests, ESLint, strict production build, browser E2E, and Ubuntu/macOS/Windows release contracts.
- [R029 · Implemented; tests passing] Add global backend status across authenticated pages (checking/online/offline/stale), last-success age, in-flight request count, manual retry, and preference-aware health polling that respects low-power mode. Browser tests verify offline-to-online retry, request loading visibility, and persistence outside the Dashboard. Full Python, lint, production build, browser E2E, and Ubuntu/macOS/Windows release checks passed in [run #44641876339](https://github.com/manit6752025/AXIOM-AI/actions/runs/44641876339).
- [R030 · Implemented; tests passing] Add bounded server-side model/dataset search, exact format/tag/favorite filters, stable sorting (including numeric parameter sizes), bounded pagination, a 5,000-entry dataset scan cap with truncation reporting, and up to 20 browser-local metadata-only model filter presets. API and Playwright tests cover query contracts, sorting/filter boundaries, pagination, and preset reload persistence. Full Python, lint, build, browser, and Ubuntu/macOS/Windows gates passed in [run #44643794865](https://github.com/manit6752025/AXIOM-AI/actions/runs/44643794865).
- [R031 · Implemented; tests passing] Add a shared accessible confirmation dialog with explicit cancel/confirm labels, safe initial focus on Cancel, explanatory scope text, Escape/backdrop handling, and busy-state locking. Migrate registry removal, saved-plan deletion, evaluation-report deletion, and workspace metadata merge/replace; remove native browser confirmation from metadata import. Browser tests verify cancellation performs no DELETE, confirmation performs exactly one, and archive replacement uses the accessible modal. Full quality gates passed in [run #44643794865](https://github.com/manit6752025/AXIOM-AI/actions/runs/44643794865).
- [R032 · Implemented; tests passing] Upgrade the global command palette to a combobox/listbox contract with `aria-activedescendant`, option position metadata, Arrow Up/Down selection, Home/End navigation, Enter execution, Escape dismissal, and a persistent polite live region announcing result counts, the selected result, and no-match state. Add keyboard/assistive-technology regression coverage and permanent accessible keyboard-help description. Full Python, lint, production build, browser E2E, and Ubuntu/macOS/Windows release gates passed in [run #44644588660](https://github.com/manit6752025/AXIOM-AI/actions/runs/44644588660).
- [R033 · In progress; validation pending] Verify responsive Control Center breakpoints, off-canvas navigation, compact top bar, single-column settings layout, and no-horizontal-overflow behavior down to 320px. Added Playwright coverage for 320–1024px phone/tablet widths, body/document overflow, and mobile navigation selection; full browser and cross-platform validation is pending.
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
- [R087 · In progress; validation pending] Report native-engine dependency readiness, supported Llama/Mistral/Qwen2/Qwen3 floating-point safetensors checkpoints, public native-model acquisition with staged installs/progress, explicit CPU/CUDA/ROCm/Metal/XPU selection, loaded backend/architecture, and local-only execution scope. Native decoding now includes validated greedy/temperature sampling, exact-count top-k filtering, nucleus top-p, probability-relative min-p, repetition penalty, reproducible sampling seeds, and stop-sequence handling, wired through the native GUI, CLI, synchronous/streaming API, unified local LLM client, and compatible Engine Fabric routes. Experimental in-RAM layer-by-layer staging and disk-resident safetensors layer streaming with CPU KV-cache offload are available with transfer telemetry. Mistral sliding-window attention now retains a compact bounded KV suffix per layer while preserving absolute RoPE positions and context limits. Grouped-query attention passes compact KV heads directly to SDPA where `enable_gqa` is supported, with expanded-SDPA and manual-math compatibility fallbacks; regression coverage compares both paths with an expanded reference. Native forward calls now validate KV-cache layer count, K/V tensor shapes, head dimensions, model dtype, memory-mode device, positive and per-layer sequence lengths before tensor operations, with malformed-cache regression coverage. Native config loading rejects boolean/coercible dimensions, model_type/architecture mismatches, malformed BOS/EOS IDs, and invalid tied-embedding flags. RoPE now supports default, linear interpolation, Llama 3 wavelength-based scaling, YaRN frequency interpolation with validated attention scaling, and dynamic NTK scaling with a factor-derived effective context bound. Regression tests cover dynamic NTK frequency math/context limits and YaRN frequency/attention scaling; these tests still require execution, and other scaling variants remain unsupported. Runtime forward inputs reject non-tensors, empty sequences, unsupported index dtypes, and prompt token IDs outside the checkpoint vocabulary before model execution. Native forward also supports fixed-length prompt microbatches and matching batched KV-cache decode, with equivalence tests against per-row execution; generation scheduling and variable-length continuous batching remain separate work. Native generate_batch() adds shared prefill/decode microbatching, per-row repetition penalties, independent EOS completion, and greedy or temperature/top-k/top-p/min-p sampling. Same-length prompt buckets are also split into bounded eight-sequence device microbatches to cap peak batch memory; responses retain input order and microbatch telemetry aggregates across chunks. This path is wired through a bounded `/api/engine/generate/batch` route, repeatable `axiom engine batch --prompt ...` and UTF-8 `--prompts-file` CLI inputs, plus a Runtime-panel batch form with per-prompt outputs and truncation/stop telemetry. Prompts with different lengths are bucketed into exact-length batches rather than naively padded; asynchronous continuous batching remains separate work. Rotary position embeddings are now computed once per forward sequence and shared by query/key projections across layers, avoiding repeated frequency-table work; checkpoint loading rejects non-floating embedding weights before selecting auto precision. Native checkpoint loading now also validates the vector dimensions of optional q/k/v/o and MLP projection biases plus an optional output-head bias before runtime execution; Qwen2 bias loading has a direct forward-pass regression test. Ordinary non-sliding prefill now delegates causal masking to SDPA without constructing a dense quadratic boolean mask, and single-token decode over a populated cache uses the all-available-key fast path; explicit masks remain for chunked and sliding-window attention. Cache and GQA changes still require full CI, release, and physical-device validation. Engine Fabric probes fixed local Ollama, llama.cpp, LM Studio, vLLM and llama-cpp-python APIs, documents their trade-offs, recommends workload profiles, and offers bounded loopback chat routing; min-p is routed only to native/Ollama adapters rather than silently ignored by other backends. It does not convert external runtime formats into native weights. AXIOM-native GGUF ingestion now supports metadata-validated single-file and split-shard dense Llama/Mistral/Qwen2/Qwen3 checkpoints and supported gguf-py tensor dequantizers, with layer-on-demand dequantization in streaming modes; Model Studio can install supported single-file GGUF directly into AXIOM's trusted native-model root or separately register it with Ollama. Native GGUF execution is floating-point dequantization, not a low-bit kernel, and tokenizer fidelity is limited to a sidecar/tokenizer JSON or supported embedded tokenizer types. EXL2/GPTQ/AWQ native low-bit kernels and broader architectures remain outstanding.
- [R088 · In progress; validation pending] Native runs expose wall time, prefill latency, cached-decode latency, time to first token, generated tokens/second and layer-streaming transfer counters. Added a bounded native benchmark in core, `POST /api/engine/benchmark`, `axiom engine benchmark`, and the existing Runtime → AXIOM Local Engine GUI. It allows 0–2 excluded warm-ups, 1–5 measured generations capped at 128 output tokens, and reports median/p95 latency and throughput without returning or persisting prompts/responses. Benchmark results now include per-run and aggregate layer-weight staging/release, disk-read, and KV-cache offload counters from measured runs only; excluded warm-ups are omitted. The GUI can cancel the request and the API signals cooperative cancellation after disconnect. Core/UI regression and browser tests still require re-execution for this telemetry addition, and physical-device validation remains outstanding.
- [R089 · Implemented; tests passing] Add prompt/token-limit inputs, excluded warm-up runs, bounded measured iterations, and explicit concurrency controls to the existing Runtime page.
- [R090 · In progress; validation pending] Provide parameter-based RAM/VRAM estimates with documented weight/KV-cache/workspace formulas, overheads, safety margin, and uncertainty range. Core, API, and Control Center paths exist; `axiom engine estimate` exposes estimates without loading weights. Runtime planning and the estimate output now distinguish Mistral's bounded retained KV cache from temporary full-context prefill memory, and both report effective context limits for dynamic NTK models. Estimates still require numerical regression execution and physical-memory calibration.
- [R091 · In progress; validation pending] Compare candidate memory estimates against supplied RAM and VRAM budgets and recommend only among currently native-supported precisions. INT4/INT8 remain sizing-only and are explicitly not loadable by AXIOM Native Engine; budget fit does not guarantee load success.
- [R092 · In progress; validation pending] Keep estimated fit separate from measured quality and speed. The Control Center presents the memory/quantization estimator and measured native benchmark in separate surfaces; benchmark results are based on observed local runs and do not use estimated fit as speed evidence. UI regression and build checks remain to be executed.
- [R093 · Implemented; tests passing] Add bounded in-memory runtime request traces with request IDs, metadata-only records, export, and explicit clear controls.
- [R094 · Implemented; tests passing] Add explicit model unload and cache cleanup, plus one-model-at-a-time cache replacement; framework allocator release remains best-effort.
- [R095 · In progress; validation pending] Add adapter compatibility checks for CPU, CUDA, ROCm, Metal/MPS, and Intel XPU; metadata/tensor-shape preflight and per-device availability reporting are wired, but real execution on each device and the release test gate remain to be validated.
- [R096 · In progress; validation pending] Auto mode already replans to CPU when accelerator memory estimates do not fit and retries CPU when native checkpoint loading runs out of memory. Native inference now also retries on CPU after an auto-selected CUDA/MPS/XPU OOM only if no token has been emitted, clears the failed runtime/cache, and reports the fallback device/reason; it deliberately refuses to replay after partial output. Regression tests cover pre-output retry and no retry after partial output, but the suite has not yet been executed; broader device-driver recovery and physical-device validation remain.

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

### 11 · Applied AI workflows and knowledge infrastructure

- [R121 · Planned] Add reproducible hardware-specific benchmark profiles with environment and provenance records.
- [R122 · Planned] Add versioned Prompt Studio templates with typed variables, validation, and revision history.
- [R123 · Planned] Add JSON Schema-constrained local inference with safe schema validation and bounded outputs.
- [R124 · Planned] Add a local embeddings engine abstraction with bounded batch and chunk processing.
- [R125 · Planned] Add local reranking adapters with deterministic score and ranking contracts.
- [R126 · Planned] Add workspace-scoped knowledge bases and safe document ingestion.
- [R127 · Planned] Add hybrid retrieval with chunking, metadata filters, source citations, and bounded search.
- [R128 · Planned] Add opt-in local multimodal image understanding with explicit model and runtime capability checks.
- [R129 · Planned] Add local speech-to-text and text-to-speech adapters with resource and privacy controls.
- [R130 · Planned] Add versioned dataset snapshots with content hashes and immutable lineage.
- [R131 · Planned] Add experiment comparison views and machine-readable statistical diffs.
- [R132 · Planned] Add local model-serving profiles with validated resource limits and readiness probes.
- [R133 · Planned] Add bounded, cancellable, resumable batch inference jobs with progress checkpoints.
- [R134 · Planned] Add request admission queues with concurrency caps, backpressure, and cancellation.
- [R135 · Planned] Add model provenance and security inspection with source, license, hash, and format evidence.
- [R136 · Planned] Add software bill of materials generation and dependency vulnerability reports with export.
- [R137 · Planned] Add versioned, default-deny plugin manifests and permission validation.
- [R138 · Planned] Add workspace backup and restore with manifests, checksums, and safe path validation.
- [R139 · In progress; validation pending] Add opt-in resource sampling and history with clear, export, and retention controls. The CLI now supports an explicit bounded watch and create-only JSONL export; the Runtime page now supports opt-in, 120-sample browser-memory history with a CPU trend, JSON export, and clear. Persistent server-side history and broader retention settings remain outstanding pending acceptance checks.
- [R140 · Planned] Publish stable Python and TypeScript SDKs verified against the supported API schema.


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

This is a proposal, not a delivery-status claim. It supplements the 140 scoped deliverables above without changing their IDs or marking unverified work as complete. Hardware validation must distinguish simulated CI coverage from measurements on physical NVIDIA, Apple Silicon, AMD, and CPU-only systems where those test environments are available. External repository metrics quoted in assessments must be checked against a current public snapshot before publication.

