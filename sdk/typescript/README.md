# AXIOM TypeScript SDK

A small TypeScript client for the versioned AXIOM HTTP API. The package works
in Node.js 18+ and modern browsers, uses the platform Fetch API, and has no
runtime dependencies.

## Install

After publishing:

```bash
npm install @axiom-ai/sdk
```

## Connect

Start the local daemon with `axiom gui`:

```ts
import { AxiomClient } from "@axiom-ai/sdk";

const axiom = new AxiomClient();

try {
  const [health, metadata, runtime] = await Promise.all([
    axiom.health(),
    axiom.meta(),
    axiom.runtime(),
  ]);
  console.log(health.status, metadata.api_contract_version, runtime);
} finally {
  axiom.close();
}
```

The default API origin is `http://127.0.0.1:40507`. Non-loopback services
must use HTTPS. The client sets `redirect: "error"` so authentication headers
are never forwarded to redirect destinations.

## Authentication and workspaces

```ts
import { AxiomAPIError, AxiomClient } from "@axiom-ai/sdk";

const axiom = new AxiomClient({
  baseUrl: "https://axiom.internal.example",
  token: process.env.AXIOM_API_READ_TOKEN,
  workspaceId: "ws-0123456789abcdef0123456789abcdef",
  timeoutMs: 30_000,
});

try {
  const models = await axiom.models({ limit: 25, offset: 0 });
  console.log(models);
} catch (error) {
  if (error instanceof AxiomAPIError) {
    console.error(error.statusCode, error.code, error.requestId);
  }
  throw error;
} finally {
  axiom.close();
}
```

The token is sent as an Authorization bearer token and is never persisted.
The workspace ID is sent as `X-AXIOM-Workspace-ID`; it selects workspace
context but is not an authorization credential. Every request carries an
`X-Request-ID`. API errors expose status, error code, request ID and details.
No automatic retry or background polling is performed.

## Workspace backup and restore

```ts
import { AxiomClient } from "@axiom-ai/sdk";

const axiom = new AxiomClient({ token: process.env.AXIOM_API_WRITE_TOKEN });
try {
  const archive = await axiom.downloadWorkspaceBackup("ws-0123456789abcdef0123456789abcdef");
  const restored = await axiom.restoreWorkspaceBackup(
    new Blob([archive], { type: "application/zip" }),
    "Restored workspace",
  );
  console.log(restored.workspace.id);
} finally {
  axiom.close();
}
```

Restore always creates a new managed workspace; it does not overwrite the
active workspace. Backup files can contain secrets, datasets and model weights.
They are not encrypted by AXIOM and must be protected by the caller.

For very large backup archives, use the streaming method instead of
buffering the complete ZIP in a single `Uint8Array`:

```ts
const stream = await axiom.downloadWorkspaceBackupStream("ws-example");
const reader = stream.getReader();
const file = await openDestinationFileSomehow();
try {
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    await file.write(chunk.value);
  }
} finally {
  reader.releaseLock();
  await file.close();
}
```

The stream owns the request timeout and abort controller until it reaches EOF,
is cancelled, or errors. Always consume or cancel it to release the request
resources. The bytes-returning method remains convenient for smaller archives.

## API v1 compatibility

The SDK targets API contract v1; this is independent of the application version.
Additive response fields are safe to ignore. Breaking changes require a parallel
API contract version. Use `axiom.meta()` to discover the contract exposed by
the running daemon. The operation manifest in the repository is checked against
the live FastAPI OpenAPI document by the SDK contract tests.
