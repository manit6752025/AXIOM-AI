# AXIOM Feature Implementation Plan

<!-- AXIOM-FEATURE-PLAN-UPDATE:START -->
## Beta implementation and acceptance criteria

The existing release has a dataset profiling foundation and model, optimisation,
quantization, workspace API and GUI components. This does not imply that every
roadmap feature is fully integrated.

For each feature, verify the following as applicable:

- **GUI:** usable controls, results, loading/error states and empty states.
- **Backend:** validated inputs, bounded resource use and useful errors.
- **CLI:** discoverable, documented commands when terminal usage is appropriate.
- **Security:** safe paths, permission checks and protected local files.
- **Tests:** unit, API, CLI and GUI checks that cover the feature.
- **Documentation:** user guide, command help, release notes and examples.

Persistent training-plan snapshots now have a bounded local store and API,
GUI history, and CLI create/list/show commands. API saves verify the displayed
snapshot against current hardware planning; stale GUI plans must be rebuilt.
CLI JSON output includes the exact persisted plan snapshot. Unversioned legacy
records migrate to schema v1 on read, while unknown future versions fail safely.
Saved workflow graphs can be listed, inspected, and structurally validated from
the CLI; this still does not execute workflow steps.
Local workflow designs have a GUI
builder, backend graph validation, versioned bounded persistence, templates,
duplication, revision history, and a planning-only dry-run with hardware estimates
and exportable reproducibility metadata; existing workflows autosave locally after
edits, but designs still cannot execute. Experiment plans have editable metadata,
tags, filtered pagination, and deletion across API, CLI, and GUI without changing
the stored plan snapshot. Dataset profiling exposes
quick, standard, and deep bounded scan budgets in the GUI and API. MCP profiles
are editable in the GUI and CLI, default-deny, and checked at each MCP tool
call. The model registry now has stable IDs with legacy migration, bounded
metadata, source/content duplicate checks, paginated API reads, safe removal,
and schema-versioned import/export. Local discovery skips symlinks, and the GUI
can show model inspection summaries. Dataset schema comparison, required-field
and type checks, and JSON report download are wired through the API and GUI.
Dataset reports now include UTF-8/newline/record diagnostics, configurable
duplicate-key thresholds, aggregate token-length bins, and sensitive-name rules
with exact-name exclusions. Profiler and API tests plus frontend lint/build pass
for these additions. Dataset PII heuristics require explicit opt-in and return
redacted counts only; provenance manifests support honest partial hashes and
optional parent-hash lineage, and a cleaning preview reports diffs without
writing files. Offline evaluation scores precomputed JSONL text without running
models, persists immutable reports with dataset and output hashes, supports
baseline regression gates, and exports value-free reports in the GUI and CLI.
Remaining priorities include workflow execution, runtime benchmark adapters,
model compatibility, measured quantization comparison, an API playground,
telemetry, and security diagnostics.
<!-- AXIOM-FEATURE-PLAN-UPDATE:END -->

1. Training workflow graph builder, versions, templates, and validation.
2. MCP agent profiles with per-tool scopes and auditable decisions.
3. Local model compatibility registry.
4. Quantization estimates separated from measured benchmarks.
5. JSONL dataset profiling and privacy indicators (included in this patch).
6. Persistent training-plan history (saved as exact, immutable plan snapshots across API, GUI and CLI), then run history.
7. Evaluation baselines and regression reports.
8. Local API playground with request limits/timeouts.
9. Historical hardware telemetry.
10. Read-only security/configuration audit.

Preserve the existing GUI theme. Do not call a feature complete until backend enforcement and tests exist.
