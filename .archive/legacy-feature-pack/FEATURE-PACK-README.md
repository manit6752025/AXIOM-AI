# AXIOM-AI roadmap feature pack

This is a **reviewable local change set**, not a claim that all 120 roadmap deliverables are complete. It adds a tested diagnostics/support slice and a live Roadmap page that reads source status labels from `ROADMAP.md`. It preserves the existing Control Center components and CSS; no style sheet is replaced or re-themed.

## Apply to your checkout

Use your existing AXIOM-AI checkout (including your own local edits). Extract the ZIP, then run:

```bash
python /path/to/axiom-roadmap-feature-pack/apply_feature_pack.py /path/to/AXIOM-AI --check-only
python /path/to/axiom-roadmap-feature-pack/apply_feature_pack.py /path/to/AXIOM-AI
```

The first command validates all edit anchors and checks that the new paths will not overwrite unrelated files. The second applies the local edits. The script refuses to continue if the repository is not a Git checkout with `ROADMAP.md`, if expected source anchors differ, or if any new file would overwrite a different existing file.

**It does not commit, push, open a PR, or call GitHub write APIs.** Review `git diff` and keep the changes on your machine until you choose to push them.

## What it adds

- Read-only workspace diagnostics through `GET /api/diagnostics/workspace`, exposed in the existing Diagnostics page and `axiom doctor --json` / `axiom status --json`.
- A privacy-minimized ZIP support bundle through `GET /api/diagnostics/support-bundle` and `axiom support-bundle`. It deliberately excludes configuration values, environment variables, credentials, datasets, model weights, logs, user records, and absolute paths.
- A validated `ROADMAP.md` reader through `GET /api/diagnostics/roadmap` and `axiom roadmap audit [--json]`.
- A new Roadmap navigation page using the current shared `FeaturePage`, `Surface`, `SummaryCard`, search, segmented filter, event-list, and status-pill patterns. It displays every parsed R001–R120 item and status labels as written in the file.
- Regression tests and user documentation for those capabilities.

The CLI audit reports source labels as-is, checks R001–R120 coverage, and reports missing, duplicate, unexpected IDs and unknown statuses. It does **not** infer that a feature is implemented merely because a similarly named file exists.

## Validation scope and limitations

See `QA-REPORT.md`. Seventeen focused tests pass in an isolated test fixture. The patcher preflight and application were tested against a mock checkout with matching source anchors, and the new Python modules plus the changed Python integration files compiled in that fixture. The actual AXIOM checkout is not mounted here, so the full repository Python suite, `npm run lint`, `npm run build`, and browser E2E tests have **not** been run against the real checkout. Do those before treating this as release-ready.

No `ROADMAP.md` status is automatically changed by this pack. Do not mark all 120 items complete based on this slice.
