# Read-only API Playground

The Control Center's API Playground is a browser UI for examining a curated list of GET endpoints. It is deliberately not a general HTTP client.

## Safety contract

- The client uses a compile-time allowlist; arbitrary URLs, methods, and request bodies cannot be supplied by the page.
- Requests use AXIOM's shared API client, which adds a correlation ID and aborts after 15 seconds.
- Inventory endpoints are requested with a small fixed page size.
- Rendered response text is capped at 100,000 characters. Prefer the bounded/paginated endpoints for large inventories.
- No new data is persisted by the Playground. Responses remain in the current component state and may contain local workspace metadata; review them before copying or sharing.
- A read-only UI is not an authorization boundary. The server must continue to enforce its own route, filesystem, and permission boundaries.

## Available endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | API process health and version |
| `GET /api/meta` | Contract metadata and compatibility policy |
| `GET /api/workspace/runtime` | Current hardware/runtime snapshot |
| `GET /api/workspace/models?offset=0&limit=5` | First small page of registered model metadata |
| `GET /api/workspace/datasets?limit=100&offset=0` | Bounded JSONL dataset metadata page; optional `path` and `search` filters; a 5,000-entry scan cap is disclosed with `truncated` |
| `GET /api/workspace/events?limit=100&offset=0` | Newest-first, paginated workspace event records; optional `search` matches action and detail |
| `GET /api/diagnostics/workspace` | Redacted read-only workspace checks |
| `GET /api/diagnostics/roadmap` | Roadmap status audit |

The Playground does not run training, inference, model downloads, mutations, or arbitrary API calls. It is diagnostic inspection only.
