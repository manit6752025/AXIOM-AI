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
