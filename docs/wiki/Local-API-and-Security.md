# Local API and Security

AXIOM's local API exposes read-only workspace information to the bundled Control Center and local integrations. The default is loopback-only.

## Start the daemon

    axiom daemon --host 127.0.0.1 --port 0

Port `0` asks the operating system for an available port. The daemon prints the selected loopback URL. Stop it with Ctrl-C.

Common read-only routes include `GET /health`, `GET /api/info`, `GET /api/summary`, `GET /api/actions`, `GET /api/hardware`, and `GET /api/tools`. The full contract is available through the running daemon's OpenAPI document.

## Network exposure

The daemon rejects a non-loopback bind unless `--allow-network` is deliberately supplied. Do not enable network access casually: review host firewall rules, trust boundaries, and any authentication/access controls for your intended deployment before exposing a service to other machines.

The daemon is not an account service, hosted AXIOM backend, or general model-inference server.

## API Playground

The beta.4 Playground is limited to a fixed GET-only allowlist, bounded request duration/reads, and capped response previews. It is not an arbitrary HTTP client and does not support state-changing methods.

## Reporting API problems

Capture the endpoint, sanitized response, HTTP status, and request/correlation ID when visible. Remove credentials, private paths, configuration values, and sensitive response data before filing an issue.

Related: [Control Center and Module Explorer](Control-Center-and-Module-Explorer), [Troubleshooting](Troubleshooting).
