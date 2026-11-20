# Local API and Security

AXIOM's local API exposes read-only workspace information to the bundled Control Center and local integrations. The default is loopback-only.

## Start the daemon

    axiom daemon --host 127.0.0.1 --port 0

Port `0` asks the operating system for an available port. The daemon prints the selected loopback URL. Stop it with Ctrl-C.

Common read-only routes include `GET /health`, `GET /api/info`, `GET /api/summary`, `GET /api/actions`, `GET /api/hardware`, and `GET /api/tools`. The full contract is available through the running daemon's OpenAPI document.

## Network exposure

The daemon rejects a non-loopback bind unless `--allow-network` is deliberately supplied. Do not enable network access casually: review host firewall rules, trust boundaries, and any authentication/access controls for your intended deployment before exposing a service to other machines.

Run `axiom security audit --json` for a read-only review of project-config
permissions, MCP policy file safety and active grants, API token-scope
configuration, OpenAI-compatible bearer-token presence, and CORS origins. The
audit never emits credential values. It confirms the built-in Control Center's
loopback bind contract and explains that the standalone daemon's optional
network mode has no authentication; it does not perform an OS port scan or
inspect external server deployments.

The optional `--daemon-host` and `--allow-network` arguments evaluate a
proposed daemon bind profile only. They do not change a running service. A
non-loopback profile is reported as unsafe because the lightweight daemon
serves local status and workspace metadata over unauthenticated HTTP.

The daemon is not an account service, hosted AXIOM backend, or general model-inference server.

## API Playground

The beta.4 Playground is limited to a fixed GET-only allowlist, bounded request duration/reads, and capped response previews. It is not an arbitrary HTTP client and does not support state-changing methods.

## Reporting API problems

Capture the endpoint, sanitized response, HTTP status, and request/correlation ID when visible. Remove credentials, private paths, configuration values, and sensitive response data before filing an issue.

Related: [Control Center and Module Explorer](Control-Center-and-Module-Explorer), [Troubleshooting](Troubleshooting).


## Workspace path boundaries

File and model selectors use workspace-relative paths. The shared path
resolver rejects absolute paths, drive-qualified paths, parent-directory
components, empty components, backslashes, control characters, and oversized
inputs. It ignores symlinked directories during discovery and rejects a
symlinked workspace root or final symlink entry.

When a selected regular file is opened on POSIX, AXIOM walks each directory
component from an open directory descriptor with `O_DIRECTORY | O_NOFOLLOW`,
then opens the final file with `O_NOFOLLOW` and verifies its descriptor is a
regular file. This prevents a symlinked parent directory from being substituted
between path selection and file opening. The resolver and opener do not follow
the final or intermediate symlink paths on this supported POSIX path.

This boundary is not a substitute for workspace ownership controls. Protect
the workspace against untrusted local writers, review external mounts and
hard-linked content, and keep file access limited to the paths and operations
the feature explicitly supports.
