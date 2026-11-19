# Storage schemas and resource identity

This document defines the compatibility contract for persisted AXIOM resources.
It applies to local workspace metadata and job history, not transient request
IDs or protocol-specific completion IDs.

## Resource IDs

New persisted resources use UUID4 values rendered as **32 lowercase
hexadecimal characters**. The canonical implementation is
`axiom.core.resource_ids.new_resource_id()`; validate with
`is_resource_id()`. IDs are unique resource identity, not sortable timestamps.
Clients must treat them as opaque strings.

- Creating a resource allocates a new ID.
- Updating a resource, renaming it, changing tags, or changing metadata keeps
  its existing ID.
- Duplicating a resource creates a new ID; copied content does not imply shared
  identity.
- Migration preserves every valid existing ID. Migration must never generate a
  replacement merely because optional metadata or a schema version is absent.
- Built-in identifiers such as the Agent Profiles `default` profile and
  workflow-template IDs are intentionally stable symbolic IDs, not UUIDs.
- Request IDs, OpenAI-compatible completion IDs, temporary filenames, and
  graph-local node IDs follow their own contracts and are not resource IDs.

### Legacy model IDs

A model registry entry from before IDs were stored may omit `id`. The registry
derives a deterministic UUID5 from the URL namespace and the exact
`axiom:model:<name>:<source>` identity string through
`legacy_resource_id("model", ...)`. The same old record therefore receives
the same ID on every read. The ID is persisted on the next explicit registry
write (for example, a model metadata update); a read alone does not rewrite
the user's file. Once written, the ID is immutable even if the model's name or
source metadata later changes. Duplicate derived IDs are rejected.

Do not change the UUID5 namespace or name construction. A future replacement
algorithm must be introduced as a new migration version and preserve already
migrated IDs.

## Persisted schema policy

Every versioned store declares its own `schema_version`; resource schema
versions are separate from the HTTP API version. Migrations are deterministic,
bounded, and performed only for explicitly supported legacy shapes. A future
or malformed version fails closed: AXIOM must not silently discard fields,
rewrite the file, or guess how to interpret data.

| Store | Current representation | Legacy behaviour | Persistence behaviour |
|---|---|---|---|
| Workflows | JSON array; each record has schema version 1, revision, and bounded versions | Missing `schema_version`, `revision`, or `versions` is read as v1 defaults; existing workflow IDs are required and preserved | A later workflow save/update/duplicate writes the v1 shape atomically |
| Experiments | JSON array; record `schema_version: 1` | Missing schema version is treated as v1; missing tags default to an empty list; future versions are rejected | A later add/update/delete writes the current v1 record shape atomically |
| Evaluation runs | JSON array; run `schema_version: 1` | Missing schema version is treated as v1; IDs must already exist and be unique; future versions are rejected | A later add/delete writes the current v1 shape atomically |
| Agent Profiles | JSON object with top-level `schema_version: 1` and a default-deny profile | No pre-v1 file migration is supported; missing, malformed, or future versions are rejected | Explicit profile writes persist the validated current schema |
| Model registry | Legacy-compatible JSON array; each model has a stable ID and validated metadata | Missing model IDs use deterministic UUID5; absent optional metadata uses dataclass defaults | Registry writes persist derived IDs; registry envelope remains an array for compatibility |
| Fine-tuning jobs | Individual job JSON records with schema version 1 | Job IDs are required; malformed records are omitted from bounded history discovery and cannot be used as resources | State transitions persist the same job ID |
| Model Studio jobs | Individual job JSON records with schema version 1 | Job IDs are required; bounded history scanning ignores invalid records | State transitions persist the same job ID |

## Safe migration rules

1. Read paths may normalise explicitly supported legacy defaults in memory.
2. A read must not rewrite the user's file. The next explicit write persists the
   migrated current-version representation.
3. Existing IDs and user-authored content are preserved byte-for-byte where a
   store's schema permits; migrations do not delete model weights, datasets,
   evaluation artifacts, or training outputs.
4. Files are validated before write and replaced atomically using the existing
   storage primitives. Unsupported future versions fail without overwriting
   the original file.
5. Adding a required field, changing field meaning, or changing ID derivation
   requires a new resource schema version, a specific migration function,
   regression fixtures for the previous version, and documented rollback
   behaviour.
6. API clients should ignore additive response fields and must not infer an
   identifier's format beyond treating it as an opaque string. Import/export
   envelopes remain independently versioned.

## Current limitations

The model registry's on-disk format is intentionally a legacy-compatible JSON
array without a top-level schema version; its API import/export envelope is
versioned separately. Agent Profiles have no older supported format to migrate.
These formats should only change through an explicit future schema migration.
