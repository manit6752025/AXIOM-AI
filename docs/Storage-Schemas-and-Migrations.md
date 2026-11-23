# Storage schemas and resource identity

This document is the compatibility contract for AXIOM's durable local metadata. It records the **actual on-disk representation**, supported legacy inputs, the event that persists a migration, and a safe rollback procedure. A response object's `schema_version` is not necessarily a persisted store version. Runtime-only and browser-memory records are identified separately below.

## Resource IDs

New persisted resources use UUID4 values rendered as **32 lowercase hexadecimal characters**. The canonical implementation is `axiom.core.resource_ids.new_resource_id()`; validate with `is_resource_id()`. IDs are opaque, stable resource identity, not sortable timestamps.

- Creating a resource allocates a new ID.
- Updating or renaming a resource keeps its existing ID.
- Duplicating a resource creates a new ID.
- Migrations preserve every valid existing ID.
- The built-in Agent Profiles `default` identifier and workflow-template IDs are intentionally stable symbolic IDs, not UUIDs.
- Request IDs, OpenAI-compatible completion IDs, temporary filenames, and graph-local node IDs follow their own contracts.

### Legacy model IDs

A model registry entry written before stable IDs were introduced may omit `id`. The registry derives deterministic UUID5 from the URL namespace and exact identity string `axiom:model:<name>:<source>` using `legacy_resource_id("model", ...)`. The same legacy entry receives the same ID on every read. A read does not rewrite the user's file; the next explicit registry write persists the derived ID. Duplicate derived IDs are rejected.

Do not change the UUID5 namespace or identity-string construction. Any future algorithm change needs an explicit schema migration and must preserve IDs already written.

## Persisted store inventory

Paths below are relative to the workspace unless explicitly marked as user-home/application-data storage. “Lazy” means AXIOM normalizes a supported legacy representation in memory, then writes the current representation only on the next explicit successful mutation. Reads alone do not rewrite user files.

| Store / path | Current on-disk schema | Supported legacy form and migration trigger | Rollback / compatibility |
|---|---|---|---|
| Workflows — `.axiom/workflows.json` | JSON array; each record has `schema_version: 2`, stable ID, revision and bounded version history | Schema-v1 records are accepted. Missing `schema_version`, `revision` or `versions` is normalized in memory (v1, revision 1, empty versions); graph shape is validated. The next explicit save/update/duplicate writes v2 atomically. | Restore a backup if rolling back to software that only understands v1. A v1 reader must not be allowed to reinterpret a v2 record or discard version history. |
| Experiments — `.axiom/experiments.json` | JSON array; each record has `schema_version: 1` | Missing schema version is treated as v1; missing tags default to an empty list. Future versions and malformed records fail closed. Explicit add/update/delete writes validated v1 records. | No format downgrade is needed for readers supporting v1. Preserve a backup before manual edits; never remove tags or IDs to force parsing. |
| Evaluation runs — `.axiom/evaluations.json` | JSON array; each run has `schema_version: 1` | Missing schema version is treated as v1. Existing run IDs must be present and unique; unsupported future versions fail closed. Explicit writes persist v1. | Preserve the original file before any manual repair. There is no automatic conversion from a different/future version. |
| Agent Profiles — `.axiom/agent-profiles.json` | Object with top-level `schema_version: 1` and a required default-deny profile | No pre-v1 migration is supported. A missing, malformed or future version is rejected; an absent file yields the default profile in memory until an explicit save. | Restore the original file or create a new default-deny store. Never synthesize grants while repairing a profile. |
| Model registry — `.axiom/models.json` | Legacy-compatible JSON array; no top-level schema envelope | Missing per-model IDs are derived deterministically with UUID5; absent optional fields use model defaults. Duplicate IDs and invalid records fail closed. The next explicit registry write persists derived IDs while retaining the array envelope. | Keep the JSON-array shape and the UUID5 namespace/identity algorithm. Do not wrap old data in a new envelope unless a separately versioned migration is implemented. |
| Integrations — `.axiom/integrations.json` | Legacy-compatible JSON array; no top-level schema envelope | Older entries may omit MCP trust/owner/connection fields; the record parser supplies safe defaults (unverified trust, unknown owner, not inspected). Explicit integration/trust/connection changes write the same array representation. | Back up before manual editing. A future envelope requires an explicit migration; preserve unrecognized user command/endpoint metadata. |
| MCP tool permissions — `.axiom/tool_permissions.json` | Unversioned object with exactly `active_profile` and `profiles`; absent file means default-deny | No versioned legacy migration exists. Unknown keys, invalid profile names/tools, malformed data or missing active profile fail closed. Only explicit profile/grant operations write the file. | Restore a known-good backup. If the format is ever versioned, the first migration must preserve default-deny and must never invent grants. |
| Security decision journal — `.axiom/security-events.json` | Top-level `schema_version: 2`, bounded hash-chained event list | Schema v1 is accepted and validated. Reads do not eagerly rewrite; the next successful append writes v2 decision metadata. The v2 records audit allow/deny outcome, tool, profile, reason and argument names (values remain redacted). | Back up the journal before rollback. A compatible v1 reader may not understand v2 decision fields; retain the backup and use a version-aware reader rather than deleting audit history. |
| Unified LLM provider profiles — `.axiom/llm-provider-profiles.json` | Object with top-level `schema_version: 2` and profiles | Schema v1 is accepted; the optional `api_key_env` field is normalized to null. The next successful profile save/delete writes v2. The field stores a validated environment-variable **name**, never a credential value. | Stop AXIOM and preserve a backup before rollback. A v1-only application can use a v1 metadata-only backup; never copy secret values into the JSON file. Provision environment variables separately. |
| Hardware snapshot history — `.axiom/hardware-snapshots.json` | Exact object with `schema_version: 1`, opt-in settings and allowlisted snapshots | There is no older schema migration. Unknown/future versions, unexpected fields and invalid fingerprints fail closed. Capture is disabled by default; explicit settings/capture writes v1. | Stop AXIOM and back up the file before removing it or rolling back. The store is optional; deleting the file resets history and opt-in settings. |
| Resource telemetry history — `.axiom/resource-telemetry.json` | Exact object with `schema_version: 1`, explicit opt-in/retention settings and sanitized resource samples | New store; no legacy migration. Persistence is disabled by default. Unknown/future versions, unexpected fields, invalid fingerprints, malformed metrics and symlinked paths fail closed. | Stop AXIOM and back up the file before rollback. Deleting it removes samples and resets the opt-in/retention settings; use the Runtime Clear history control to remove samples while preserving settings. |
| Workspace catalog — `~/.axiom/workspaces.json` (or `AXIOM_WORKSPACES_HOME`) | Object with `schema_version: 1`, default workspace ID and workspace records | A missing catalog is initialized with the discovered default workspace. Existing malformed or non-v1 catalogs fail closed; there is no automatic pre-v1 migration. New managed workspace directories are data roots, not schema files. | Back up the catalog and managed workspace directories together. Do not hand-edit root paths or IDs; restore a known-good catalog before reusing a managed workspace. |
| Prediction jobs — `.axiom/prediction_jobs.json` | Object with `schema_version: 3` and bounded `jobs` array | Schema-v1 records migrate with empty checkpoint/fingerprint metadata and remain non-resumable. Schema-v2 records migrate by hashing their saved generation settings. Schema-v3 validates dataset/model/adapter digests, a generation-settings digest (sampling parameters and row-field mapping), checkpoint filename/inode/size/SHA-256, row counters and resumption count. Unknown versions, malformed records and duplicate IDs fail closed. State transitions replace the bounded file atomically (descriptor-relative on supported POSIX systems). | Keep a backup before rollback. Job history is coordination/provenance data; never resume by editing its status, runner identity or checkpoint metadata. Use the explicit `axiom engine prediction resume JOB_ID` command or GUI/API action so AXIOM verifies all inputs and checkpoint bytes. |
| Fine-tuning jobs — `$AXIOM_FINETUNE_HOME/jobs/<id>.json` or `~/.local/share/axiom/fine-tuning/jobs/<id>.json` | Per-job JSON record written with `schema_version: 1`; restrictive directory/file permissions | Job IDs are required. There is no declared older-version migration; direct reads require a valid JSON object and the bounded discovery path skips invalid records. Job checkpoints are separate training artifacts, not migrations of the job JSON record. | Preserve the job JSON and output/checkpoint directory as a unit. Do not downgrade by changing a job's version or status; use AXIOM's recovery/diagnostic paths. |
| Model Studio jobs — workspace `.axiom/model-studio/jobs/<id>.json` or `$AXIOM_MODEL_STUDIO_HOME/jobs/<id>.json` | Per-job JSON record written with `schema_version: 1` | Bounded discovery skips invalid JSON/IDs; there is no declared historical migration. Current readers validate the basic object/ID; the Model Studio reader does not yet enforce a strict future-version gate. | Back up the job record together with any referenced downloaded/conversion output. Do not manually replay a job by changing its status. |
| Workspace metadata transfer — exported ZIP | Separate archive envelope: `manifest.json` with format `application/vnd.axiom.workspace-metadata+zip` and `schema_version: 1` | This is an import/export archive format, separate from each collection's schema. The allowlist currently contains models, experiments, workflows, evaluations and Agent Profiles. Import validates the manifest, checksums, collection versions and bounds before mutation. | Keep the original archive. Restore/import only with a version that recognizes the archive envelope and each included collection schema. Configuration, credentials, MCP grants, integrations, prediction jobs, runtime traces, datasets and weights are intentionally excluded. |
| Saved plan snapshots / software manifests | Embedded in experiment records; plan snapshot and software manifest use `schema_version: 1` | Not a separate top-level file. Validation is performed when comparing/rendering the saved plan. It requires complete captured plan/hardware inputs; it does not silently regenerate a missing snapshot. | Preserve the parent experiment record. Future embedded snapshot changes must version their own payload and retain the historical captured estimate. |

### Non-persisted and derived data (not migration stores)

- MCP inventory/handshake and schema-browsing objects are bounded API/CLI projections; connection state itself is stored in the unversioned integrations array.
- Runtime traces are in-memory only (bounded ring buffer); they are not restored after process restart.
- Engine point-in-time telemetry is generated on demand. Browser telemetry samples are opt-in and browser-memory-only (maximum 120); R113 benchmark-linked samples are also browser-memory-only (maximum 20).
- R114 JSON/CSV exports are local browser downloads, not AXIOM server-side persistence.
- API response schema versions describe an API contract, not an on-disk migration.
- `axiom.yaml` is project configuration rather than a versioned metadata-store envelope; diagnostics validate parsing and permissions but do not rewrite it as a migration.

## Safe migration rules

1. A store owns its schema version. Do not infer a version from a neighbouring store, the HTTP API version, or a package version.
2. A read may normalize only a specifically documented legacy shape. It must not write the normalized result back merely because it was read.
3. The first successful explicit mutation after a lazy migration writes the current schema atomically and preserves valid resource IDs and user-authored values.
4. Unsupported future versions, malformed records and unknown fields fail closed unless the store explicitly documents an additive extension point. Never drop unknown persisted data to make a file parse.
5. Every breaking field/meaning change requires: a new version, deterministic migration function, previous-version fixture, tests for malformed/future schemas, a documented trigger (read vs next write), and rollback guidance.
6. Rollback is restore-from-backup, not a reverse migration unless an explicit reverse migration is implemented. Preserve related artifacts (job outputs, model registry entries, workspace roots) with the metadata that references them.
7. Back up private/secret-bearing metadata securely. Never include API keys, access tokens, environment values, raw user prompts/responses or model/dataset contents in schema fixtures or migration diagnostics.

## Current limitations

The model registry and integration registry intentionally retain legacy JSON-array envelopes. MCP tool permissions have no explicit version field yet and must remain default-deny. Fine-tuning and Model Studio job files write schema-v1 records but currently lack a full historical migration framework; Model Studio's reader does not yet reject every future version explicitly. These are documented limitations, not claims of forward migration support. They must be addressed before either store's schema is changed.

## Unified LLM provider profiles — schema 1 to 2

The workspace-local `.axiom/llm-provider-profiles.json` document advances from schema version 1 to version 2 to add optional `api_key_env`. The field is either null or a validated environment-variable name; it must never contain a credential value. Existing schema-v1 documents remain readable without eager rewrite. The next successful profile save/delete writes schema v2 and adds `api_key_env: null` to legacy profiles. Unknown fields, embedded keys and malformed references fail closed.

Rollback: stop AXIOM, preserve a backup, then restore a version-1 metadata-only profile file if rolling back to software that only understands schema 1. Do not copy credential values into the file. Environment variables are provisioned separately and are not exported by profile backup or API responses.

## Hardware snapshot history — schema 1

The workspace-local `.axiom/hardware-snapshots.json` document uses schema version 1 and contains explicit opt-in settings plus up to 100 allowlisted hardware captures. Capture is disabled by default. Reads do not probe hardware and no periodic sampler is started. Retention is bounded by both count and age. Unknown/future schemas and unexpected fields fail closed.

Rollback: stop AXIOM and back up the file before removing it. Deleting the file resets snapshot history and opt-in settings. Hardware snapshots are optional and are not required for runtime startup, model access or inference.
