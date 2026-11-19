# AXIOM Roadmap

<!-- AXIOM-ROADMAP-UPDATE:START -->
## Beta roadmap and implementation status

### Current foundations

- [x] Bounded JSONL profiling with row, field, privacy, UTF-8/newline, duplicate-key, token-length, opt-in PII, manifest, and cleaning-preview diagnostics.
- [x] Workspace API foundations for datasets, runtime information and training plans.
- [x] Existing model inspection, registry, optimisation and quantization components.
- [x] GUI sections for the principal engineering workflows.

### Planned feature priorities

| Priority | Feature | Goal |
|---|---|---|
| 1 | Visual training workflow builder | Connect and validate training steps visually. |
| 2 | Agent profiles and tool permissions | Add explicit permissions and auditable tool usage. |
| 3 | Model registry and compatibility | Improve discovery, metadata and hardware/runtime checks. |
| 4 | Quantization comparison lab | Compare options with documented assumptions. |
| 5 | Dataset profiling and privacy checks | Expand bounded validation and privacy signals. |
| 6 | Persistent experiment history | Save settings, outcomes and run history. |
| 7 | Evaluation and regression reports | Repeat evaluations and compare results. |
| 8 | Local API playground | Exercise supported endpoints and inspect responses. |
| 9 | Hardware telemetry and resource history | Record resource snapshots and performance trends. |
| 10 | Security and configuration audit | Provide actionable local diagnostics. |

### Delivery standard

A feature should not be marked shipped until the applicable backend, GUI,
CLI, validation, tests and documentation have been verified. Roadmap entries
remain planned work until that evidence exists.
<!-- AXIOM-ROADMAP-UPDATE:END -->

## v0.3.0-beta.1
- [x] JSONL dataset profiler foundation, local API, CLI and existing Dataset-page integration.
- [ ] Visual training workflow builder.
- [ ] Backend-enforced MCP profiles and permissions.
- [ ] Rich model compatibility scoring.
- [ ] Persistent experiment history and evaluation regressions.
- [ ] Local inference playground and hardware telemetry history.
- [ ] Comprehensive security/configuration audit.

## Distribution gates
GitHub Releases and PyPI remain primary. npm, APT, Homebrew, Scoop/WinGet, AUR and GHCR require verified artifacts, checksums, configured credentials/accounts and successful workflow smoke tests. WinGet/AUR may require human submission. Do not announce publication before workflows succeed.
