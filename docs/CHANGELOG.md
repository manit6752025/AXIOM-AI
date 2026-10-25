# Changelog

## 0.3.0-beta.5 — Beta candidate

### Added and improved
- Added workspace diagnostics and privacy-minimized support bundles through the CLI, API, and Control Center.
- Added a roadmap audit command and GUI integrity reporting to make delivery status easier to verify.
- Improved saved workflow design with templates, revision history, duplication, validation, and dry-run resource estimates.
- Improved API Playground safety with a fixed read-only GET allowlist, bounded reads, request timeouts, and capped previews.
- Continued hardening native inference controls, cancellation, runtime metrics, and CLI/API/GUI diagnostics.

### Release status
- Source metadata is aligned to 0.3.0-beta.5. This entry describes the intended candidate scope, not proof that every feature or platform has passed validation.
- Publish only after the release workflow, platform installer smoke tests, interactive GUI checks, full test suite, and package/tag version checks succeed.
- The npm package is @netcore-technologies/axiom-cli; its version is tracked separately from the private axiom-ui development package.

## Unreleased — safer configuration guidance

- Add stable configuration validation error codes, bounded YAML shape checks, and non-mutating repair suggestions.
- Regression-test that malformed, missing, oversized, and symlinked configs are not echoed or modified by validation.


## Unreleased — environment, compatibility, and retention

- Allow `AXIOM_GUI_PORT` to set the default local Control Center port; explicit `--port` remains authoritative and invalid values fail clearly.
- Add a dependency-free compatibility report for Python, Node.js, and validated OS targets, run in cross-platform CI.
- Add a data-retention guide and a Settings shortcut to review and clear the in-memory runtime trace buffer.
- Update delivery status for native engine discovery, runtime benchmarking, trace lifecycle, connector discovery, API Playground, and release smoke coverage after the existing end-to-end checks passed.


## Unreleased — latest/development

### Added
- Add native LoRA adapter compatibility preflight to the Runtime GUI and API: inspect target modules and tensor shapes without loading full base weights, and report CPU/CUDA/ROCm/MPS availability separately.
- Added cooperative cancellation for native inference streams: browser disconnects are monitored asynchronously, cancellation reaches the AXIOM decoder, and tests cover cancellation before prompt prefill and between tokens.
- Added explicit workspace-root resolution with environment override and nearest-project discovery, plus bounded read-only workspace diagnostics and safe support bundles exposed through CLI and API.
- Added metadata-only discovery of trusted native model/LoRA directories, newline-delimited token streaming with Stop/partial-output retention in the GUI, and `axiom engine discover [--json]` / `axiom engine run --stream` in the CLI.
- Added versioned `--json` output for model/integration inventory, Hugging Face metadata inspection, local model inspection/search, and dataset inspect/validate/stats; dataset commands now share one JSON schema and preserve validation exit codes.
- Added schema-versioned JSON output to `axiom dataset clean`, including per-category removal counts and source-preservation semantics; added model registry export/import/remove/favorite commands with duplicate-batch checks and live-registry overwrite protection.
- Exposed architecture, license, repeatable tags, content SHA-256 and validated metadata updates through `axiom model add/update`; failed metadata edits leave the existing record unchanged.
- Added native GGUF installation from Model Studio and the CLI's `install-gguf --native` target, plus direct AXIOM-owned GGUF parsing/dequantization for supported single-file and complete split-shard dense Llama/Mistral/Qwen2/Qwen3 checkpoints. Source quantization remains input storage only; the runtime uses floating-point tensors and does not claim low-bit memory/speed benefits.

- Added AXIOM-owned Llama/Mistral inference execution with custom RMSNorm, RoPE, grouped-query attention, KV cache, supported LoRA deltas, sampling and autoregressive decode over local safetensors; the inference path does not call Transformers model.generate or Ollama.
- Constrained direct AXIOM Engine model/adapter paths to configured trusted roots (including post-resolution symlink checks), replaced path-derived API names with names from validated directories, and added explicit model unload/resource cleanup.
- Added a bounded in-memory runtime trace history with request IDs for local generation and benchmarking, safe error codes, metadata-only JSON export, and explicit clear controls; prompts and generated responses are not recorded.
- Added measured local Ollama inference benchmarking to the existing Runtime page and loopback API, with warm-ups, bounded measured iterations/concurrency, request latency percentiles, token-throughput coverage, and no benchmark-result persistence.
- Added server-side model registry search, exact format/tag filters, favorites-only selection, numeric parameter sorting, filtered pagination, and browser-local saved filter presets in the existing Models page.
- Added a model-specific hardware-fit report in the existing Models page: metadata-derived weight/runtime memory ranges, RAM/VRAM budget screening, format-family guidance, and explicit warnings when quantization or GPU-memory evidence is missing.
- Added the Community and Adoption Strategy covering reproducible benchmarks, hardware calibration, optional integrations, real-world validation, and contributor outreach.
- Added AXIOM Engine, an AXIOM-owned direct local causal-LM runtime with Control Center and CLI access, bounded generation, and optional local PEFT adapter loading.
- Added AXIOM's own small-batch PyTorch LoRA training loop, masked padding labels, finite-loss checks, gradient clipping, adapter artifact verification, and dependency version diagnostics.

### Fixed

- Aligned LoRA optional dependency minimums with Transformers 5 and blocked training when installed dependency versions are too old.
- Improved local model/adapter loading errors without enabling remote model code or model downloads.

### Scope

- AXIOM owns the engine orchestration and training loop; the pretrained base weights are user-supplied local model files and remain subject to their original licences.

## 0.3.0-beta.4

### Added

- Added bounded native stop sequences across Runtime GUI, API and CLI; split-token stop matching is held back in streaming output, stop text is excluded from the final response, and finish reason distinguishes EOS, stop string, token limit and cancellation.
- Added measured native prefill latency, cached decode time, time-to-first-token and completion tokens/second to Runtime results, with API typing, UI display and tiny-checkpoint/E2E assertions.
- Exposed temperature, top-p, top-k, repetition penalty and optional seed in the native Runtime form; typed API streaming carries every setting into AXIOM's decoder and reports effective sampling metadata.
- Added a searchable Control Center Module Explorer across GUI pages, CLI command families, and API surfaces.
- Added workspace diagnostics with machine-readable CLI output using `axiom doctor --json`.
- Added a roadmap audit command, `axiom roadmap audit --json`, and GUI roadmap integrity reporting.
- Added privacy-minimized support archive generation with `axiom support-bundle`.
- Expanded the read-only API Playground with a fixed GET-only allowlist, request timeouts, bounded reads, and capped response previews.
- Expanded saved workflow design tooling with templates, revision history, duplication, validation, and dry-run resource estimates.

### Changed

- Improved local Control Center page navigation and administrator remembered-session behaviour.
- Improved request/error diagnostics and discoverability across the local CLI, API, and GUI.
- Updated the README release badge and highlights for beta.4.

### Scope and execution boundary

- Workflow validation and dry-run estimates do not execute training.
- Evaluation scores precomputed predictions; it does not run model inference.
- Runtime discovery reports locally detected candidates and readiness evidence; it does not claim a runtime is connected when it is not.
- No hosted workspace or cloud inference service is introduced by this release.


## 0.2.0-beta.9

### Added

- Read-only daemon routes for the local workspace: `/api/models`, `/api/files`,
  `/api/model/inspect`, `/api/dataset/inspect`, `/api/train/plan`,
  `/api/ai/plan`, and `/api/headroom`.
- The daemon serves the bundled web UI at `/ui/` when `axiom/webui` is present.
- Quantization Lab comparison table showing FP16, INT8, and INT4 weight sizes
  against detected VRAM or RAM. Sizes are rough estimates, not benchmarks.
- Automatic PyPI publishing on version tags, with a check that the tag,
  `pyproject.toml`, and `axiom/version.py` agree before anything is uploaded.

### Changed

- Windows, Linux, macOS, and PyPI release builds now build the web UI and
  bundle it into every package format.
- Optimizer, Quantization Lab, and Beta.5 panels use the shared app theme
  tokens, so they match the rest of the UI in dark and light mode.

### Security

- Daemon workspace paths must resolve inside the working directory.
- The daemon refuses requests with a non-loopback `Host` header unless it was
  started with `--allow-network`.

### Fixed

- CodeQL alerts for modules imported with both `import` and `import from` in
  the daemon and product contract tests.
- Local `npm run build` failures caused by a missing `rolldown` native binding
  after an incomplete optional-dependency install.

## 0.2.0-beta.8 — release preparation

This section describes the beta.8 target. It is not a claim that release
assets, workflows, or hosted pages have passed live verification.

### Documentation and product contract

- Rewrote the first-minute CLI path around `axiom`, `guide`, `summary`, the
  model and dataset checks, hardware planning, and `tools doctor`.
- Documented the loopback-only daemon, operating-system port selection, and
  explicit network opt-in.
- Corrected examples for positional model metadata and dataset-clean output
  arguments.
- Separated shipped inspection/planning behavior from future training,
  inference, telemetry, and hosted-workspace work.
- Kept Linux bundle and Python fallback installation behavior explicit.
- Documented Headroom as an optional external context-compression proxy that
  AXIOM may explain or discover but never bundles or starts.
- Added an explicit Headroom status and passthrough path that never installs
  or starts the external proxy during discovery.
- Hardened Linux release installation with pinned tags, user-scoped paths,
  archive validation, and checksum verification.

## 0.2.0-beta.7

See the [published beta.7 release](https://github.com/NetCore-Technologies/AXIOM-AI/releases/tag/v0.2.0-beta.7)
for the assets and commit associated with that earlier prerelease.

## 0.2.0-beta.6

- Focused the Linux distribution on a reusable unpacked tarball for repeated
  terminal launches.
- Added checksum verification to the Linux bundle installer.
- Published Linux DEB, AppImage, one-file ELF, tarball, and installer assets
  when the release workflow completed successfully.
- Smoke-tested the version command, loopback daemon health route, bundle
  contents, DEB metadata, and installer syntax in CI.

## 0.2.0-beta.5

- Added optimizer and quantization planning foundations.
- Added filesystem-boundary hardening and expanded release validation.
- Documented the status of unsigned, unnotarized macOS distribution assets.
