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
  workspace knowledge bases, plugin-manifest inventory, and local prediction
  job start/list/status/resume/cancel.

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

The gated publisher is `.github/workflows/sdk-release.yml`. It is
`workflow_dispatch` only and runs the contract checks, Python SDK tests,
TypeScript build/tests, and package builds before any publish job starts. It
accepts `both`, `python`, or `typescript`; ordinary pushes never publish.

Before publishing, configure a PyPI Trusted Publisher for repository
`manit6752025/AXIOM-AI`, workflow `sdk-release.yml`, environment `pypi`.
Create the GitHub `pypi` deployment environment. For npm, configure the
`npm` deployment environment and add a package-scoped publish token as the
`NPM_TOKEN` repository/environment secret. The workflow uses npm provenance
and publishes `@axiom-ai/sdk` with public access.

The workflow does not run until a maintainer explicitly dispatches it, and
registry credentials are never committed to the repository. The packages
remain unpublished until the external PyPI/npm publisher configuration and
release action are completed successfully.

## Local prediction jobs

Both SDKs expose typed prediction-job lifecycle methods. Explicit start and
resume calls consume local compute; list/status calls return bounded metadata
only. Resume revalidates the dataset fingerprint, model/adapter, generation
settings, and checkpoint identity and digest on the server. A stale or damaged
checkpoint is rejected instead of being published or silently restarted.
