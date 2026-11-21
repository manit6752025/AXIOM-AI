# AXIOM SDKs

AXIOM publishes independently installable clients for its versioned local API.

| Language | Package | Runtime | Install |
| --- | --- | --- | --- |
| Python | `axiom-ai-sdk` | Python 3.11+, standard library only | `python -m pip install axiom-ai-sdk` |
| TypeScript | `@axiom-ai/sdk` | Node.js 18+ or modern browsers, Fetch API | `npm install @axiom-ai/sdk` |

The package source lives in `sdk/python` and `sdk/typescript`. They are
versioned independently from the Control Center bundle, but currently target
the AXIOM API contract v1.

## Common client guarantees

- Explicit base URL, bearer token, workspace selector, per-request timeout,
  request IDs and structured API errors.
- Non-loopback origins must use HTTPS. Redirects are not followed, so bearer
  credentials are not forwarded to another origin.
- No implicit retries, cloud-provider calls, telemetry sampling or background
  polling.
- Workspace backup download and restore methods; restore creates a new managed
  workspace and does not overwrite the active workspace.
- Typed methods for metadata, health, workspace inventory, models, datasets,
  local runtime, engine status/telemetry, persistent telemetry history,
  workspace knowledge bases and plugin-manifest inventory.

The Python SDK uses the standard library. It provides both a bytes-returning
backup convenience method and a chunked method that writes to a
caller-owned binary stream. The TypeScript SDK provides a bytes convenience
method and a streaming `ReadableStream` method.

## API compatibility

`GET /api/meta` reports `api_contract_version` separately from the AXIOM
application version. The SDK operation registry at
`sdk/openapi/axiom-v1-operations.json` is checked against the live FastAPI
OpenAPI schema. The verifier also ensures that each listed public operation is
implemented in both packages:

```bash
python scripts/verify_sdk_contract.py --json
python -m pytest -q tests/test_sdk_contract.py tests/test_sdk_python.py
npm install --prefix sdk/typescript
npm test --prefix sdk/typescript
```

Additive response fields are supported within v1. Breaking changes require a
parallel API version; clients should ignore unknown response properties.

## Publishing

The Python and TypeScript packages have independent package metadata and
release versions. Build and inspect them separately:

```bash
python -m pip install build
python -m build --sdist --wheel --outdir dist sdk/python
npm install --prefix sdk/typescript
npm pack --prefix sdk/typescript
```

Publish only after the SDK quality workflow and full repository quality gates
pass. The package metadata is prepared for PyPI and the public npm registry;
credentials and the actual publish action remain release-environment duties.
