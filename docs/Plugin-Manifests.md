# Declarative plugin manifests and permission validation

AXIOM supports workspace-local, versioned JSON manifests for describing plugin
identity, tool metadata, and requested permission scopes. This is an inventory
and authorization-policy layer; it is **not a plugin execution runtime**.

## Manifest schema

A manifest has exactly these top-level fields:

- `manifest_schema_version`: integer `1`.
- `id`: unique lowercase reverse-DNS style identifier, for example
  `com.example.research`.
- `name`, `version`, and `description`: display metadata; version must use
  semantic versioning.
- `tools`: one to 32 unique tool declarations. Each declaration contains only
  `name`, `description`, and a non-empty set of requested `scopes`.

Supported scopes are `read`, `write`, `execution`, `network`, and
`secrets`. The grant UI and CLI let an operator select a subset of each tool's
declared scopes. A tool cannot receive a scope it did not declare.

Unknown fields are rejected. Manifests cannot provide `command`,
`entrypoint`, executable paths, scripts, URLs, import names, or lifecycle
hooks. AXIOM stores and validates the declaration; it does not import plugin
modules, launch commands from manifests, or invoke the described tools.
External MCP servers remain a separate integration type with their own
unverified-by-default trust control and read-only schema inspection.

The published JSON Schema is
[`docs/schemas/plugin-manifest-v1.schema.json`](schemas/plugin-manifest-v1.schema.json).
An example is at `examples/plugin-manifest.example.json`.

## Version and grant lifecycle

Manifests are stored in the workspace's `.axiom/plugins/registry.json`.
Registration starts in `default_deny`; it never creates a grant. An update
must use a strictly newer semantic version. Each normalized manifest has a
canonical SHA-256 fingerprint.

Permission grants bind to that exact fingerprint and contain explicit
tool/scope pairs. Any manifest update changes its fingerprint; old grants are
marked `stale_manifest` and cannot authorize the new declaration. Operators
must inspect the new version and submit a new explicit grant. Revoke removes
all grants while retaining the registered metadata.

Manifest registration, grants, and revocations are recorded in the
workspace's hash-chained security event journal. If the event journal cannot
record a successful mutation, the CLI/API reports that the policy change may
already have taken effect and asks the operator to inspect the registry before
retrying.

## CLI

```bash
axiom plugin manifest validate ./plugin-manifest.json
axiom plugin manifest validate ./plugin-manifest.json --json
axiom plugin manifest register ./plugin-manifest.json
axiom plugin list
axiom plugin list --json
axiom plugin grant com.example.research \
  --manifest-sha256 <current-sha256> \
  --tool search:read \
  --tool annotate:read,write
axiom plugin revoke com.example.research
```

## API

- `GET /api/security/plugins` lists manifests and permission state.
- `POST /api/security/plugins/manifests` validates/registers a manifest.
- `PUT /api/security/plugins/{plugin_id}/permissions` grants selected
  tool/scope pairs against the supplied current manifest SHA-256.
- `DELETE /api/security/plugins/{plugin_id}/permissions` revokes all grants.

All API payloads reject unknown fields; manifest versions and grants are
validated server-side. The server is authoritative—the GUI does not grant
capabilities by itself.

## Resource and security limits

A workspace stores at most 50 manifests, each at most 64 KiB, with at most 32
tool declarations per manifest. The combined registry is capped at 4 MiB.
Identifiers, descriptions, semantic versions, scope values, and unique tool
names are validated. Registry and lock symlink paths are rejected, malformed
registries fail closed, and updates are lock-protected and atomic. On POSIX,
the metadata directory, lock, registry reads, and replacement writes are
anchored through no-follow directory/file descriptors to reduce path-swap
races. Other platforms use the platform's file-lock and safe-file primitives.
No manifest content is treated as executable code.

A permission grant in this metadata store does not by itself authorize an
MCP built-in tool or any operating-system action. Built-in MCP tools continue
to use the independent server-side `ToolPermissionStore` and must be granted
there. No plugin adapter, sandbox, subprocess lifecycle, or execution
capability is enabled by this manifest feature.
