# Control Center and Module Explorer

Launch the local browser interface with:

    axiom gui

Use `axiom gui --no-open` to print the local URL without opening a browser.

## Module Explorer

The beta.4 Module Explorer is a searchable index of current GUI pages, CLI command families, and API surfaces described by the live OpenAPI reference. GUI destinations navigate to an existing page; CLI-only entries provide copyable command guidance rather than pretending a browser can execute arbitrary shell commands.

The Explorer is a discoverability layer. It does not make planned roadmap items available or prove a backend capability exists just because it appears in a catalog.

## Areas to explore

| Area | Purpose | Boundary |
| --- | --- | --- |
| Workspace overview | Summarize project and environment information. | Visibility depends on the active workspace and process permissions. |
| Models | Inspect local files and available metadata. | Inspection is not inference or proof of model compatibility. |
| Datasets | Validate/profile supported dataset structure. | Structural checks do not guarantee semantic quality or compliance. |
| Optimizer and Quantization Lab | Compare conservative sizing estimates and draft plans. | Estimates are not benchmark results. |
| Diagnostics | Report workspace health checks. | Checks are read-only; they do not auto-repair problems. |
| Roadmap | Display recorded roadmap status and integrity results. | Status is not inferred from file presence. |
| API Playground | Send bounded requests to predefined read-only endpoints. | No arbitrary URL entry or mutating methods. |
| Workflow design | Manage templates, revisions, validation, and dry-run estimates. | Designs do not execute pipelines. |

## Quick troubleshooting

1. Confirm the intended build with `axiom version`.
2. Run `axiom doctor --json` and inspect failed check IDs.
3. Restart `axiom gui` from the intended workspace.
4. Record the visible request/correlation ID for API failures; redact private paths and environment details.
5. Inspect any support archive before sharing it.

Related: [Getting Started](Getting-Started), [Diagnostics](Workspace-Diagnostics-and-Support-Bundles), [Local API and Security](Local-API-and-Security).
