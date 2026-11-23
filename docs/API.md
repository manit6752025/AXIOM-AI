# AXIOM Local API

The local daemon publishes its OpenAPI contract at `/api/openapi.json`, with
interactive documentation at `/api/docs` and `/api/redoc`. `GET /api/meta`
reports the API contract version separately from the AXIOM application version.
Clients should send a unique `X-Request-ID` header; the server validates and
echoes it on success and failure responses. If the value is missing or malformed,
the server generates a safe identifier. The browser client includes that ID in
visible request errors, and the CLI daemon readiness probe sends one as well.

## Compatibility policy

- The current contract is `v1`; existing resource paths remain under `/api`.
- Additive, backward-compatible fields and endpoints may be added within v1.
- Existing field meanings, validation guarantees, and side-effect boundaries do
  not change within v1.
- Breaking changes require a parallel `/api/v2` contract. The v1 contract remains
  available during a documented migration window; removal requires a later major
  application release and a release note.
- Clients should ignore unknown response fields and must not assume that an
  optional field is present unless the OpenAPI schema marks it required.
- API version does not replace per-resource `schema_version` fields used for
  persisted workspace data.
- Persisted resource ID and workspace schema migration rules are defined in
  [Storage schemas and migrations](Storage-Schemas-and-Migrations.md).

## API permission scopes

API scope enforcement is opt-in so existing loopback-only Control Center and CLI
workflows remain compatible. To enable it for API clients, configure **both**
environment variables with distinct, high-entropy values:

- `AXIOM_API_READ_TOKEN`: permits `GET`/`HEAD` requests and the explicitly
  allowlisted validation/planning `POST` endpoints.
- `AXIOM_API_WRITE_TOKEN`: permits read access and all write, job, execution,
  cancellation, model-load, and other side-effecting operations.

Generate each value independently, for example with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`. Tokens must be
32–256 ASCII characters, and the read and write values must differ. AXIOM rejects
a partial, weak, or duplicate configuration at startup rather than silently
running with incomplete scope enforcement.

Read-only `POST` operations are deliberately allowlisted: workflow validation
and dry-run, model optimization/policy inspection, optimizer planning, workspace
training-plan and dataset previews, model inspection, and engine compatibility
or quantization estimates. Every other `POST`, plus `PUT`, `PATCH`, and
`DELETE`, requires the write token. New mutation routes therefore fail closed
until explicitly classified. A read token cannot create or update a saved
workflow, start/cancel a job, pull or unload a model, generate inference, or
invoke a remote provider.

Send credentials as `Authorization: Bearer <token>`. Missing/invalid credentials
return HTTP 401; a valid read token used for a write operation returns HTTP 403.
Both use the standard AXIOM error envelope and preserve `X-Request-ID`. Health,
metadata, and OpenAPI documentation endpoints remain public. The OpenAI-compatible
`/v1/models` endpoint requires read scope, and `/v1/chat/completions` requires
write scope. The bundled same-origin Control Center remains usable; cross-origin
API clients must provide a token. When both variables are unset, token enforcement is disabled for
backwards compatibility. Keep the daemon on loopback for ordinary local use;
tokens do not replace TLS, network isolation, or host security.

## AXIOM Engine and training

- `GET /api/engine/status` reports direct local inference readiness.
- `POST /api/engine/generate` runs inference from an existing local Transformers model directory and optionally a local PEFT adapter.
- `GET /api/fine-tuning/runtime` reports compatible local LoRA training dependencies.
- `POST /api/fine-tuning/jobs` starts one bounded local LoRA training job.

Engine and fine-tuning model loads are local-only; they do not download weights or execute remote model code. See [AXIOM Engine](Local-Engine.md) and [Local fine-tuning](Local-Fine-Tuning.md) for limits and supported inputs.

The API binds to loopback by default through `axiom gui`. Network binding is an
explicit operator choice; see the local-daemon security guidance before exposing
it beyond the machine.


## Unified LLM client and mobile bridge

- `GET /api/llm/providers` lists local/cloud adapters without requiring credentials or contacting a provider.
- `POST /api/llm/models` lists models for the selected provider. Cloud provider requests require a key and can perform a network request.
- `POST /api/llm/chat` runs one bounded text completion. Cloud requests require `consent: true`; otherwise the API rejects the request before the provider call.

The `ollama-local` provider uses the existing local inference adapter, which talks only to `127.0.0.1:11434` and uses Ollama's structured chat API. It does not require cloud consent or credentials. Use `axiom-local` for AXIOM's native safetensors engine; these local runtimes have distinct model-format support.
- `GET /v1/models` and `POST /v1/chat/completions` expose AXIOM Native Engine models using the OpenAI-compatible schema.

The shared LLM API uses request-scoped credentials. It does not persist provider keys, prompts, or generated answers to the AXIOM workspace. Cloud requests are sent over HTTPS to fixed provider hosts including OpenAI, Anthropic, Gemini, DeepSeek, Moonshot/Kimi, and OpenRouter; custom OpenAI-compatible base URLs must pass the public-HTTPS validator. Request sizes and output tokens are bounded.

Run `axiom mobile-bridge` to start a separate app that mounts only the two `/v1` endpoints for AXIOM-native models and locally installed Ollama models. Ollama model IDs are namespaced as `ollama/<model-name>` in `/v1/models` to avoid collisions with native model IDs. The bridge routes them only to `127.0.0.1:11434`, never to a remote inference provider. It generates a random bearer token for the process. Use `--listen-lan` only on a trusted private network: the bridge uses plain HTTP rather than TLS, and its token does not protect against a network observer able to read traffic. Never port-forward it to the public internet; use a VPN or encrypted tunnel when crossing an untrusted network. The general `axiom gui` Control Center remains bound to loopback.

See [Unified LLM client and mobile bridge](Unified-LLM-Client.md) for examples and the supported-provider limits.


## Collection pagination

`GET /api/workflows` uses a bounded cursor page (`limit` 1–50, optional `q` search, and `after_workflow_id` from the previous page's `next_cursor`). The cursor is tied to the same search filter; restart the listing to include newer saved designs.


Workspace collection endpoints use bounded `limit` and `offset` parameters and return `total`, `limit`, `offset`, and `has_more` where paging applies.

- `GET /api/agent-profiles`: bounded profile pages (`limit` 1–100, default 100; `offset` 0–100), retaining the capability catalog and authorization note on every page.
- `GET /api/fine-tuning/jobs`: newest-first pages (`limit` 1–100, default 20; `offset` 0–5,000). Discovery scans at most 5,000 newest job records and returns `truncated` and `scan_limit` when the inventory exceeds that cap.
- `GET /api/models/studio/jobs`: bounded, searchable newest-first pages (`limit` 1–100, `offset` 0–5,000, optional `search`), with explicit truncation metadata.

- `GET /api/workspace/models`: bounded model registry pages; search, exact format/tag filters, favorites, and stable sorting are applied before pagination.
- `GET /api/workspace/experiments`: bounded experiment pages; query and tag filters are applied before pagination.
- `GET /api/workspace/datasets`: `limit` 1–500 (default 500), `offset` 0–5000, optional `path` and case-insensitive `search`. Discovery is capped at 5,000 JSONL entries; `truncated: true` explicitly indicates that the inventory exceeded the scan cap. Results are sorted by workspace-relative path before paging.
- `GET /api/workspace/events`: `limit` 1–100 (default 100), `offset` 0–100, optional case-insensitive `search` across action and detail. Results remain newest-first and reflect the bounded in-memory event buffer.

The frontend typed client keeps the existing array-returning `datasets()` and `events()` methods for compatibility and adds `datasetsPage()` and `eventsPage()` for callers that need pagination metadata.

## Environment configuration

AXIOM reads configuration from the current process environment. The Control
Center remains bound to IPv4 loopback; this is deliberate and cannot be loosened
by setting an environment variable.

| Variable | Purpose | Default / behavior |
| --- | --- | --- |
| `AXIOM_GUI_PORT` | Default port for `axiom gui` / `axiom ui` | `40507`; an explicit `--port` wins; `0` selects an ephemeral port |
| `AXIOM_CORS_ORIGINS` | Comma-separated exact browser origins allowed by FastAPI CORS | Local Vite origins on ports `4173` and `5173` |
| `AXIOM_WORKSPACE` | Workspace-root override used by workspace resolution | If unset, nearest-config/current-directory resolution applies |
| `AXIOM_MODEL_ROOT` | Default local model root for daemon/model inspection compatibility | Workspace-local `models` directory where the daemon sets it |
| `AXIOM_MODEL_ROOTS` | Additional trusted roots for native local model and adapter discovery | No additional roots |
| `AXIOM_PYTHON` | Python executable selected by the JavaScript GUI launcher | `python` on Windows; `python3` elsewhere |

Do not add a public origin to `AXIOM_CORS_ORIGINS` unless you understand the
browser trust implications. CORS is not authentication. Keep the service on
loopback for ordinary local use; network exposure requires a separately
reviewed deployment and access-control design.


## Model Studio job cancellation and timeout

`POST /api/models/studio/jobs/{job_id}/cancel` requests cooperative cancellation of
an active Model Studio job. The response returns the updated job with status
`cancelling`; the worker reports terminal status `cancelled` after it reaches a
safe boundary. Completed, failed, cancelled, and timed-out jobs are returned
unchanged. Missing IDs return HTTP 404 and malformed IDs return HTTP 422.

Downloads check cancellation between 1 MiB chunks and remove partial temporary
files before returning. Ollama/llama.cpp subprocesses are terminated and then
killed if they do not exit within the cleanup grace period. Model Studio jobs
have a four-hour hard deadline, individual remote reads have bounded network
timeouts, and subprocesses have operation-specific timeouts. Deadline expiry
returns the distinct terminal status `timed_out`. Cancellation cannot undo an
external side effect that has already committed (for example, a completed model
registration); such work is reported as completed rather than falsely marked
cancelled. If cancellation is requested during Ollama registration after the GGUF
file is downloaded, that complete file remains in its selected local destination;
AXIOM does not delete a successful download while cancelling a later registration
step.


## Reproducible saved plan snapshots

- `GET /api/workspace/experiments/{experiment_id}/reproducible`: return the deterministic canonical JSON document for a saved plan. The response includes the captured inputs, hardware snapshot, exact saved estimate, planner provenance, execution-not-started marker, and SHA-256 fingerprint.
- `GET /api/workspace/experiments/{left_id}/diff/{right_id}`: compare two saved snapshots and return a sorted field-level diff, categorized as inputs, hardware, planner provenance, estimate, or format. Missing experiments return HTTP 404; incomplete or invalid saved snapshots return HTTP 422.
- New experiment records store `planner_algorithm_version` and `axiom_version`. Existing snapshots without those fields remain readable, but the reproducible document reports unknown provenance rather than guessing.

The fingerprint intentionally excludes display names, notes, tags, record IDs, and timestamps. It covers the model/dataset labels, parameter count, requested method, captured hardware, planner provenance, and exact plan estimate. Snapshot printing and diffing are read-only: neither reruns the planner nor creates a job or artifact. A matching fingerprint is not a guarantee of identical runtime behavior or training results.

## Workspace metadata transfer

- `GET /api/workspace/metadata/export`: returns a versioned ZIP archive with media type `application/vnd.axiom.workspace-metadata+zip`. The archive contains the model registry, workflow designs, experiment plans, evaluation reports, and agent task profiles.
- `POST /api/workspace/metadata/import`: accepts the raw ZIP bytes with `Content-Type: application/vnd.axiom.workspace-metadata+zip` (also accepts `application/zip` or `application/octet-stream`). It validates the archive manifest, member allowlist, size limits, SHA-256 checksums, schemas, and record conflicts before applying changes.
- The default import mode merges records and rejects duplicate IDs, model names/sources, or agent-profile IDs/names. `?replace=true` replaces only collections included in the archive; omitted collections remain unchanged.
- The response reports per-collection imported counts, mode, and excluded categories. Invalid archives return HTTP 422; uploads over 24 MiB return HTTP 413 and unsupported media types return HTTP 415.

The archive is not a complete workspace backup. It excludes `axiom.yaml`, credentials, MCP permission grants, external integration commands and URLs, prediction-job history, runtime traces, datasets, model weights, checkpoints, generated artifacts, and temporary files. Some included metadata can still contain local filesystem paths; review archives before sharing. Imports never extract arbitrary ZIP paths and use staged, restrictive-permission writes with rollback on commit failure.


## Security event journal

The workspace-local security journal records only allowlisted MCP permission
and profile changes. It does not store token values, prompts, commands, profile
contents, or user-supplied event summaries. The event list is bounded to 2,000
records (including review acknowledgements) and a 5 MiB JSON file.

- `GET /api/security/events?limit=50&offset=0`: lists up to 100 recent action events, newest first, with their review state and chain-integrity summary. Review acknowledgements are attached to their target action and are not counted as separate actions in the list.
- `GET /api/security/events/verify`: verifies the complete retained hash chain.
- `GET /api/security/events/export`: returns the complete redacted JSON journal and its head hash for external retention.
- `POST /api/security/events/{event_id}/review`: appends one review acknowledgement. The JSON body is `{"reviewer_label":"Security Admin"}`; labels are restricted to 2–48 safe characters and a given event can be reviewed only once.

The event-list, verification, and export endpoints use the API read scope. Review
is a mutation and therefore requires the API write scope under the normal
request-scope middleware. In the bundled Control Center, the same-origin
administrator session follows the existing local API policy.

Reviewer labels are **self-attested**, not verified account identities. The
journal's SHA-256 chain detects partial edits, deletion, and accidental
reordering, but it is not a digital signature: a fully privileged local user
who can rewrite the entire journal can recompute the chain. For independent
tamper resistance, export the journal to storage controlled separately from
the AXIOM workspace. At capacity, AXIOM refuses further entries instead of
silently rotating or deleting history; export and archive the journal before
restarting event capture.
