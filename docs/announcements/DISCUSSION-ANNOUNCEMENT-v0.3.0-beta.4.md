# 🚀 AXIOM v0.3.0-beta.4 — better discovery, diagnostics, and workflow design

I'm sharing the beta.4 update for AXIOM AI. This release focuses on making the local-first toolkit easier to navigate, diagnose, validate, and support—across the CLI and the Control Center GUI.

## ✨ What's new

- **Module Explorer:** search the current GUI pages, CLI command families, and API surfaces from one place.
- **Workspace Diagnostics:** run `axiom doctor` or `axiom doctor --json` to inspect workspace configuration, filesystem access, disk space, Python compatibility, and required dependencies.
- **Roadmap Audit:** use `axiom roadmap audit --json` to check the scoped 120-item roadmap for missing, duplicate, unexpected, or unknown IDs/statuses. The audit reports integrity findings; it does not silently rewrite roadmap statuses.
- **Support Bundles:** create a minimized diagnostic archive with `axiom support-bundle --output ./axiom-support-bundle.zip`. Please inspect it before sharing.
- **Safer API Playground:** predefined GET-only endpoints, bounded requests, timeouts, and capped response previews—rather than arbitrary URLs or state-changing methods.
- **Workflow Design Improvements:** templates, versioned saves, duplication, revision history, structural validation, and dry-run resource estimates.
- **Control Center Polish:** improved local navigation, remembered administrator-session behavior, and clearer API/request diagnostics.

## 🔒 Still local-first—and explicit about limits

AXIOM helps you inspect models and datasets, understand the hardware visible on your system, validate a workspace, and draft plans before you commit resources. Workflow designs and dry runs are not training jobs. Evaluation scores precomputed predictions and does not run inference. Runtime discovery does not mean a runtime is installed or connected, and this release does not introduce a hosted workspace or cloud inference service.

I'd rather clearly label a limitation than claim a feature works when it hasn't been implemented and verified.

## 🧪 Help test beta.4

Please try the CLI and Control Center in your environment and share reproducible bugs or usability feedback. Testing across Windows, Linux, and macOS is especially useful. When reporting an issue, include the AXIOM version, OS/Python version, the command or GUI path, and sanitized diagnostics. Don't attach credentials, private data, model weights, or an unreviewed support bundle.

**Release page:** https://github.com/NetCore-Technologies/AXIOM-AI/releases/tag/v0.3.0-beta.4  
**Release workflow:** https://github.com/NetCore-Technologies/AXIOM-AI/actions/workflows/release-assets.yml  
**Repository:** https://github.com/NetCore-Technologies/AXIOM-AI  
**Changelog:** https://github.com/NetCore-Technologies/AXIOM-AI/blob/main/docs/CHANGELOG.md

> Beta note: please check the release page and linked Actions run for the actual downloadable packages and their status. A tag by itself does not confirm that all platform assets have been published.

Thanks to everyone following AXIOM and helping improve it.

**AXIOM — Build AI. Own AI.**
