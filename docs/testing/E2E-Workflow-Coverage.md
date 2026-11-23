# End-to-end workflow coverage inventory

The machine-readable mapping in `e2e-workflow-coverage.json` maps every Control Center navigation route declared by `src/App.tsx:navItems` to the Playwright specifications and acceptance journeys that exercise it.

The mapping covers all 18 registered page IDs: optimizer, quantizer, dashboard, models, datasets, training, workflows, evaluation, runtime, MCP inspector, diagnostics, API Playground, agent profiles, Prompt Studio, roadmap, module explorer, logs, and settings.

The CI regression `tests/test_e2e_coverage_inventory.py` checks that:

- Every registered navigation page has an inventory entry, with no stale or duplicate IDs.
- Each entry documents at least one acceptance journey and references an existing Playwright specification.
- Every referenced specification contains a Playwright test declaration.

This is an inventory-integrity check, not a claim that every conceivable workflow or state transition is covered. Runtime has multiple dedicated specs for high-risk local inference, serving, cancellation, telemetry, knowledge retrieval, speech, vision, and job flows. The inventory should be updated alongside any new navigation page or high-risk journey.
