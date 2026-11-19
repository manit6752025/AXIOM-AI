# AXIOM v0.3.0-beta.4 release notes

Beta.4 emphasizes discoverability, read-only diagnostics, workflow-design management, and clearer operational boundaries.

## Added

- A searchable Control Center Module Explorer for current GUI pages, CLI command families, and API surfaces.
- Workspace diagnostics with `axiom doctor --json` for machine-readable results.
- Roadmap integrity auditing with `axiom roadmap audit --json` across the scoped 120-item roadmap.
- Privacy-minimized support archive generation with `axiom support-bundle`.
- A bounded API Playground using a fixed GET-only allowlist, request timeouts, bounded reads, and capped response previews.
- Workflow-design templates, revision history, duplication, validation, and dry-run resource estimates.

## Changed

- Improved local Control Center navigation and remembered administrator-session behavior.
- Improved API error/request diagnostics and feature discoverability across the CLI, API, and GUI.
- Refreshed README highlights and the beta.4 changelog entry.

## Important product boundaries

- Workflow validation and dry-run estimates do not execute training.
- Evaluation scores predictions already present in the input; it does not run inference.
- Runtime discovery reports candidates/readiness evidence; it does not assert a runtime is connected when it is not.
- No hosted workspace or cloud inference service is introduced by this release.

## Verify before downloading

A release tag does not guarantee successful cross-platform publication. Check the [v0.3.0-beta.4 release page](https://github.com/NetCore-Technologies/AXIOM-AI/releases/tag/v0.3.0-beta.4) and the [release workflow](https://github.com/NetCore-Technologies/AXIOM-AI/actions/workflows/release-assets.yml) for final status, assets, and checksums.

Related: [Changelog](https://github.com/NetCore-Technologies/AXIOM-AI/blob/main/docs/CHANGELOG.md), [Getting Started](Getting-Started).
