# Phase 2 scope and roadmap mapping

## Added, wired, and locally tested

| Feature | Backend/API | GUI | CLI / docs | QA in this environment | Roadmap mapping |
|---|---|---|---|---|---|
| Read-only API Playground | Uses a closed client allowlist and current typed API routes; GET only | New page reuses current nav/page-header/theme system | Endpoint catalog, safety docs and release docs | E2E assertions authored; TS/TSX syntax parsed; full Playwright not run here | R109; partial R110/R120 |
| Runtime connector discovery | New passive inventory function and `GET /api/diagnostics/runtime-connectors`; router registered in API app | New panel embedded into Runtime page | Discovery contract documented | Core + API unit tests; TS/TSX syntax parsed; full GUI suite not run here | Partial R085, R096, R117, R120 |
| Versioned Agent Profiles | Bounded local JSON store, schema validation, atomic write, symlink checks, CRUD API | New GUI page with create/edit/version/delete | `axiom agent-profile list/create/set-capabilities/delete`; docs | Core, API and CLI focused tests | Partial R097; does not complete R099 or replace MCP policy |
| Release smoke checker | Checks project/runtime version, roadmap IDs/statuses, router wiring and typed client endpoints | Checks GUI Page union/nav/pageMeta/render switch and styling | Shell runner and README | Unit test plus mock-checkout execution | Supports R012/R117/R118/R120 |
| E2E auth setup | Not applicable | Reusable Playwright helper exercises welcome/setup/sign-in using UI | No test bypass or seeded password hashes | Spec syntax checked; browser execution still required | Supports R120 |

## Explicit non-claims

This is not completion of all 91 roadmap items still open. Inference/training is not executed by this pack. Runtime discovery is passive presence detection, not endpoint connection, compatibility negotiation, load/unload, performance measurement or recovery. Agent Profiles are task-intent metadata, not enforcement scopes; existing MCP server-side permissions remain authoritative. Cross-platform CI static checks are not the same as testing released installer artifacts on each OS. Roadmap status labels are left unchanged.

The full repository pytest suite, ESLint, TypeScript production build, and Playwright E2E suite were not run in this environment because a full local checkout with its Node dependencies was not available. The workflow added in the pack runs these gates in GitHub Actions after application.
