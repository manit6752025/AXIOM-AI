# AXIOM Python SDK

The Python SDK is a small, synchronous client for the versioned AXIOM HTTP API.
It uses Python's standard library and requires Python 3.11 or newer; no HTTP
client dependency is added to an application.

## Install

After publishing:

```bash
python -m pip install axiom-ai-sdk
```

For local development from this repository:

```bash
python -m pip install ./sdk/python
```

## Connect

Start the local API with `axiom gui`, then:

```python
from axiom_sdk import AxiomClient

with AxiomClient() as axiom:
    print(axiom.health())
    print(axiom.meta())
    runtime = axiom.runtime()
    print(runtime["status"])
    print(axiom.workspaces()["workspaces"])
```

The default origin is `http://127.0.0.1:40507`. Remote non-loopback origins
must use HTTPS. The client does not follow redirects, preventing bearer
credentials from being forwarded to another origin.

## Authentication and workspace selection

```python
from axiom_sdk import AxiomClient

client = AxiomClient(
    "https://axiom.internal.example",
    token="YOUR_AXIOM_API_READ_OR_WRITE_TOKEN",
    workspace_id="ws-0123456789abcdef0123456789abcdef",
    timeout=30,
)
try:
    models = client.models(limit=25, offset=0)
finally:
    client.close()
```

Configure `AXIOM_API_READ_TOKEN` and `AXIOM_API_WRITE_TOKEN` in the daemon
before using scoped bearer access. Use the read token for read operations and
the write token only for operations that mutate workspace state or perform
runtime actions. The SDK never persists credentials. The workspace ID is
transmitted as `X-AXIOM-Workspace-ID`; it is a routing selector, not an
authorization credential.

Every request carries a generated `X-Request-ID`. `AxiomAPIError` exposes
`status_code`, `code`, `request_id`, and structured `details`. Transport
and timeout failures are represented by `AxiomNetworkError`.

## Examples

```python
from axiom_sdk import AxiomClient, AxiomAPIError

with AxiomClient(token="READ_TOKEN") as axiom:
    for model in axiom.models(limit=20)["models"]:
        print(model["name"])

    print(axiom.engine_telemetry()["cpu"])
    print(axiom.telemetry_history(limit=120)["total"])

    try:
        axiom.configure_telemetry_history(enabled=True, max_samples=120, max_age_days=30)
        axiom.record_telemetry_sample()
    except AxiomAPIError as exc:
        print(exc.status_code, exc.code, exc.request_id)
```

Workspace backups are returned as bytes. Restores always create a new managed
workspace and do not overwrite the active workspace:

```python
from pathlib import Path
from axiom_sdk import AxiomClient

with AxiomClient(token="WRITE_TOKEN") as axiom:
    archive = axiom.download_workspace_backup("ws-0123456789abcdef0123456789abcdef")
    Path("workspace-backup.zip").write_bytes(archive)
    result = axiom.restore_workspace_backup(archive, "Restored workspace")
    print(result["workspace"]["id"])
```

Backups may contain private workspace files, datasets, model weights and
credentials saved in the workspace. Archives are not encrypted; protect them.

## Compatibility

The package supports API contract v1. The API contract version is independent
of the AXIOM application version. Clients should ignore additive response
fields; breaking API changes require a parallel API version. Use `axiom.meta()`
to check compatibility before relying on newer endpoints.

The SDK has no automatic retries, implicit cloud-provider calls or background
polling. Sampling, mutations, model loading and generation happen only when
their explicit methods are called.
