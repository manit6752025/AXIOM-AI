# Control Center and Module Explorer

Launch the local browser interface with:

    axiom gui

Use `axiom gui --no-open` to print the local URL without opening a browser.

## Global command palette

Press **Ctrl+K** (or **⌘K** on macOS) from any authenticated workspace page
to search the Control Center without leaving the keyboard. Results combine
workspace pages with actions such as switching the dark/light theme, changing
the administrator password, and signing out. Use **↑/↓** to move the selected
result, **Enter** to run it, and **Escape** to close the palette. The action
list reflects the current theme so it always offers the opposite theme as the
switch target.

The palette is a local navigation/action surface. It does not search workspace
records, execute model or workflow jobs, or expose commands that require
additional destructive confirmation.

## Global loading and backend status

The authenticated Control Center top bar displays the local API connection
state on every workspace page. It shows **Checking**, **Online**, **Offline**,
or **Stale**, the age of the last successful response when relevant, and a
manual **Retry backend connection** control. Requests made through the typed
JSON API client contribute to a shared in-flight loading count so users can
see when workspace data is being refreshed.

Health checks run when the authenticated shell opens and then follow the
Dashboard refresh preference; low-power mode disables periodic checks.
Manual retry remains available. A successful response from any typed JSON
request refreshes the last-success timestamp. The stale label means no
successful response has been observed for more than one minute; it is a
connectivity freshness hint, not a claim that individual cached records have
expired. Streaming requests are not included in the shared request counter.

## Responsive phone and tablet layouts

The Control Center shell supports narrow phone widths down to 320 CSS pixels,
uses a collapsible off-canvas navigation drawer at 780 pixels and below, and
reflows page headers, forms, status strips, and settings grids for small
screens. On phone layouts the top-bar search collapses to an icon until focused;
the profile name and secondary backend status detail are compacted to keep the
primary navigation and connection retry control reachable. The drawer closes
after a page is selected. Automated browser checks exercise 320px, 360px,
390px, 480px, and tablet widths, and check that the document does not introduce
horizontal overflow. This is responsive regression coverage, not certification
for every device or browser.

## Deep links and browser history

Control Center pages support direct links using the `page` query parameter,
for example `/?page=models`, `/?page=training`, `/?page=settings`, and
`/?page=api-playground`. A direct link opens the selected page after the local
administrator session is restored. Sidebar, command-palette, and in-page
navigation update the URL with browser history entries; **Back** and
**Forward** restore both the page content and the selected navigation item.
Unknown page slugs fall back to the Control Center overview.

The URL represents the current browser view only. It does not include
workspace credentials, filesystem paths, or private model/dataset contents.

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
