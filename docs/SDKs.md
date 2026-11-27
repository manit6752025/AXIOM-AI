# AXIOM Python and TypeScript SDKs

AXIOM exposes a versioned HTTP contract under `/api`. The independently
installable Python and TypeScript SDKs provide typed access to the local API
without importing the React Control Center or requiring AXIOM internals.

## Packages

- Python: [`axiom-ai-sdk`](../sdk/python), Python 3.11+, standard-library
  HTTP transport, no runtime dependencies.
- TypeScript: [`@axiom-ai/sdk`](../sdk/typescript), Node.js 18+ and modern
  browsers, platform Fetch API, no runtime dependencies.

Install examples:

```bash
python -m pip install axiom-ai-sdk
npm install @axiom-ai/sdk
```

For repository development, use `python -m pip install ./sdk/python` and
`npm install --prefix sdk/typescript`.

## Request and security contract

Both clients support an explicit API origin, optional bearer token, optional
workspace selector, timeout, caller headers, and request IDs. Every request
includes a generated `X-Request-ID`; when supplied, the active workspace is
sent as `X-AXIOM-Workspace-ID`.

Loopback HTTP is supported for local development. A non-loopback host must
use HTTPS. Python refuses redirects and TypeScript uses Fetch's
`redirect: "error"`, ensuring a bearer token is never forwarded to a redirect
destination. Tokens are held only in the client instance and are not stored
on disk. Do not give a write token to read-only integrations.

API errors are surfaced as structured `AxiomAPIError` objects with HTTP
status, stable error code, request ID, and details. Network failures and
timeouts use a separate network-error class. Neither SDK retries failed
requests or changes request semantics behind the caller's back.

The workspace ID header is a routing/context selector, not authentication.
Configure `AXIOM_API_READ_TOKEN` and `AXIOM_API_WRITE_TOKEN` in the daemon
when bearer-scope enforcement is required. Read tokens are for safe read
operations; operations that mutate files or execute runtime actions need the
write scope.

## Representative operations

Both clients expose:
- Health, API metadata and OpenAPI document.
- Workspace runtime, model inventory, dataset inventory and workspace catalog.
- Workspace creation and rename.
- Engine status and point-in-time telemetry.
- Explicitly enabled persistent telemetry history settings, record, list and
  clear operations.
- Knowledge-base list/create/search and plugin-manifest inventory.
- Read-only workspace diagnostics, scoped-roadmap audit, bounded privacy-filtered runtime traces, security posture audit and SPDX 2.3 SBOM inventory.
- Full workspace backup download and restore as a new workspace.
- Local prediction-job start, list, status, cooperative cancellation and verified restart-resume.

They do not start telemetry sampling, run model generation, install models,
call cloud providers, or alter workspace state unless the caller invokes the
corresponding operation.

### Python

```python
from axiom_sdk import AxiomClient

with AxiomClient(token="READ_TOKEN", workspace_id="ws-example") as axiom:
    metadata = axiom.meta()
    models = axiom.models(limit=20, search="qwen")
    telemetry = axiom.engine_telemetry()
    print(metadata["api_contract_version"], len(models["models"]), telemetry["sampled_at"])
```

The Python client offers `download_workspace_backup_to(workspace_id, output)`
to copy a large ZIP archive in fixed-size chunks into a caller-owned binary
stream, avoiding a full archive-sized Python bytes object.

### TypeScript

```ts
import { AxiomClient } from "@axiom-ai/sdk";

const axiom = new AxiomClient({
  token: process.env.AXIOM_API_READ_TOKEN,
  workspaceId: "ws-example",
});
try {
  const [metadata, models] = await Promise.all([
    axiom.meta(),
    axiom.models({ limit: 20, search: "qwen" }),
  ]);
  console.log(metadata.api_contract_version, models);
} finally {
  axiom.close();
}
```

For very large archives, `downloadWorkspaceBackupStream(workspaceId)`
returns a standard `ReadableStream<Uint8Array>`. Consume it or cancel it so
the associated request timeout and response resources are released.

## Backup and restore

Backup archives can include project configuration, workspace metadata,
datasets, model weights and private files or credentials stored in the
workspace. They are not encrypted by AXIOM; keep them protected.

Restores validate archive paths and per-file checksums and register a new
managed workspace only after successful verification. They do not overwrite
the active workspace:

```python
with AxiomClient(token="WRITE_TOKEN") as axiom:
    archive = axiom.download_workspace_backup("ws-source")
    restored = axiom.restore_workspace_backup(archive, "Restored workspace")
```

## Versioning and verification

The SDK releases target API contract v1. Application version and API contract
version are distinct. API v1 allows additive fields; breaking changes require
a parallel API version and a migration window. Call `meta()` before relying
on a newly introduced operation.

A maintainer can publish the independently versioned packages through the
manual `Publish AXIOM SDK packages` workflow. It validates and builds both
packages before the selected publish jobs run. PyPI must have a Trusted Publisher configured for this repository/workflow and
the `pypi` environment. npm must have a trusted publisher configured for
repository `manit6752025/AXIOM-AI`, workflow `sdk-release.yml`, and
environment `npm`; the workflow uses npm's OIDC trusted-publishing support
and provenance, without storing a long-lived npm token in the repository.
The first creation of the npm package is an owner-controlled bootstrap step
because npm requires the package to exist before its trusted publisher can be
registered. Use the `validate-only` dispatch option to build both packages,
install their artifacts in clean environments, and verify imports without
publishing anything. The `both`, `python`, and `typescript` options publish
only after the validation/build job succeeds. Pushes and pull requests do not
publish packages.

The 30-operation registry in
[`sdk/openapi/axiom-v1-operations.json`](../sdk/openapi/axiom-v1-operations.json)
is validated against the application's live FastAPI OpenAPI document and both
SDK method declarations by `scripts/verify_sdk_contract.py`.

Run the SDK checks:

```bash
python scripts/verify_sdk_contract.py --json
python -m pytest -q tests/test_sdk_contract.py tests/test_sdk_python.py
npm install --prefix sdk/typescript
npm test --prefix sdk/typescript
```

The repository CI runs these checks independently of frontend bundling. SDK
package publication remains a separate release action after tests and quality
gates pass.
