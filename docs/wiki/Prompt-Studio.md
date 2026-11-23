# Prompt Studio

Prompt Studio stores reusable prompt templates in the active AXIOM workspace. Templates are validated metadata, not executable workflows.

## Safety and limits

- Rendering performs literal `{{variable_name}}` substitution only. It does not evaluate expressions, run tools, start a model, or contact a provider.
- Variable values must match their declared scalar type. Strings, integers, finite numbers, booleans, and enumerated choices are supported.
- Unknown values, missing required values, undefined placeholders, malformed placeholder expressions, and out-of-range values are rejected.
- The workspace store is bounded to 100 templates and 8 MiB. A body is limited to 64 KiB, a render to 128 KiB, and each template to 32 typed variables.
- Updates require the current `expected_version`. A stale save or delete returns a conflict instead of overwriting a newer edit.
- Up to 20 prior revisions are retained per template. Each revision records a SHA-256 content fingerprint.
- Storage is workspace-local, created with restrictive permissions on POSIX systems, and protected by a file lock and atomic writes.

Template metadata and rendered output can contain user-authored prompt content. Review the text before sending it to any model or external provider. Prompt Studio itself never sends it.

## Control Center

Open **Prompt Studio** from the sidebar or use the global command palette. Create a template, edit its body and typed variable definitions, save a new revision, and preview it with JSON scalar values. Revision history can be inspected and loaded for editing. Saving a loaded historical revision creates a new version; it does not rewrite retained history.

## CLI

List, inspect, create, update, render, and delete templates locally:

```bash
axiom prompt list --json
axiom prompt create "Research brief" \
  --description "A short structured brief" \
  --body "Write a {{tone}} brief about {{topic}}." \
  --variables-json '[{"name":"tone","type":"choice","choices":["formal","friendly"]},{"name":"topic","type":"string","required":true}]' \
  --json
axiom prompt show <template-id> --json
axiom prompt versions <template-id> --json
axiom prompt render <template-id> \
  --values-json '{"tone":"friendly","topic":"local inference"}' --json
axiom prompt update <template-id> --expected-version 1 \
  --name "Research brief v2" --body "Prepare a {{tone}} brief about {{topic}}." \
  --variables-json '[{"name":"tone","type":"choice","choices":["formal","friendly"]},{"name":"topic","type":"string","required":true}]'
axiom prompt delete <template-id> --expected-version 2 --yes
```

For larger bodies, use `--body-file PATH`. The CLI rejects symlinked body files and invalid UTF-8. Deletion requires both the expected current version and explicit `--yes`.

Supported variable definitions:

```json
[
  {"name":"topic","type":"string","required":true,"min_length":1,"max_length":120},
  {"name":"count","type":"integer","required":false,"default":3,"minimum":1,"maximum":20},
  {"name":"temperature","type":"number","required":false,"default":0.2,"minimum":0,"maximum":2},
  {"name":"include_sources","type":"boolean","required":false,"default":true},
  {"name":"tone","type":"choice","required":true,"choices":["formal","friendly"]}
]
```

Only placeholders with the exact `{{variable_name}}` form are accepted. Substitution is not a template language and does not support filters, expressions, includes, or code execution.

## API

The workspace API exposes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/prompt-templates` | Bounded list with search and offset pagination |
| POST | `/api/prompt-templates` | Create version 1 |
| GET | `/api/prompt-templates/{id}` | Read a template |
| PUT | `/api/prompt-templates/{id}` | Update with `expected_version` |
| GET | `/api/prompt-templates/{id}/versions` | List retained revisions |
| GET | `/api/prompt-templates/{id}/versions/{version}` | Read a retained revision |
| POST | `/api/prompt-templates/{id}/render` | Validate values and render literal substitutions |
| DELETE | `/api/prompt-templates/{id}?expected_version=N` | Delete with optimistic version check |

Validation errors return 422, stale version conflicts return 409, missing templates return 404, and unsafe storage failures return a generic 500 response. Mutating routes require write scope when API bearer-token scopes are configured.

## Related references

- [CLI reference](CLI-Reference.md)
- [API reference](../API.md)
- [AXIOM roadmap](../../ROADMAP.md), deliverable R122
