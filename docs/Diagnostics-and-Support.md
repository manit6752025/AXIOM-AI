# Workspace diagnostics and support bundles

AXIOM's Diagnostics page and CLI use the same read-only workspace checks. The
checks cover workspace discovery, `axiom.yaml` parsing, unsafe symlink entries,
filesystem read/write access, free disk space, Python compatibility, and the
required Python package imports and installed distribution versions. Dependency
minimums are checked against the runtime requirements in `pyproject.toml`.
On POSIX systems, a project configuration writable by group or other local
users is reported as a failure. Results are structured with stable IDs so the
GUI, scripts, and tests can interpret them without parsing human-oriented text.

## Commands

```bash
axiom doctor
axiom doctor --json
axiom doctor --strict --json
axiom status --json
axiom roadmap audit
axiom roadmap audit --json
axiom support-bundle --output ./axiom-support-bundle.zip
axiom support-bundle --output ./axiom-support-bundle.zip --force --json
```

`AXIOM_WORKSPACE=/path/to/project` selects a workspace. `--workspace` explicitly
overrides it for `doctor` and `support-bundle`. Discovery otherwise selects the
nearest parent containing a regular `axiom.yaml`, then the current directory.
Caller-provided workspace symlinks are rejected.

The doctor checks the active Python interpreter rather than another Python
binary found on `PATH`. It verifies that every required runtime module is
importable and that its installed distribution metadata meets the declared
minimum version. Missing modules/metadata and outdated distributions are
reported separately in the JSON `workspace.summary` fields
`missing_dependencies` and `outdated_dependencies`. Dependency versions are
also recorded in the support bundle's environment manifest.

On POSIX systems, `axiom.yaml` must not be group- or other-writable. This is a
tamper-resistance check: diagnostics do not read, print, or export configuration
values. Use `axiom doctor --strict --json` in CI when workspace warnings should
also produce a non-zero exit status; ordinary `--json` keeps warnings
informational for projects that have not created every optional directory.


## Roadmap audit

`axiom roadmap audit` and the Control Center Roadmap page parse the scoped
120-item section from `ROADMAP.md`. The audit reports the source labels without
inferring completion from file existence, validates R001–R120 coverage, and
surfaces duplicate IDs, missing IDs, unexpected IDs, and unknown statuses.
It is read-only and does not change roadmap labels. A roadmap entry should only
be marked implemented after its applicable backend, GUI, CLI, regression tests,
and documentation gates have been verified in a real checkout.

## Runtime trace history

The Diagnostics API exposes `GET /api/diagnostics/runtime-traces?limit=50&offset=0`.
Use optional `search` to match only the stored request ID, operation, model,
status, or error code. Pages are newest-first and return filtered `total`,
`has_more`, `offset`, and `limit` metadata. The buffer is still capped at
200 entries in process memory; when the capacity is reached, older records may
have rotated out. Prompt and response bodies are never retained, and clearing
the buffer remains an explicit action.

## Support-bundle privacy contract

The ZIP contains only `diagnostics.json`, `environment.json`, and a static
`README.txt`. It does not include configuration file contents, environment
variables, credentials, full/absolute paths, model weights, datasets, logs,
experiment records, or user-authored data. The bundle includes aggregate check
results, app/Python versions, OS family and required dependency versions. This
is a minimization measure, not a guarantee that every metadata combination is
anonymous: inspect the archive before sharing it.

The HTTP bundle endpoint sends `Cache-Control: no-store`, a fixed download name,
and `X-Content-Type-Options: nosniff`. The CLI refuses to overwrite an existing
file unless `--force` is provided and rejects output symlinks. No diagnostics
or bundle generation step writes into the workspace.

## Scope and status

This feature pack adds a diagnostics API and GUI section, a live roadmap view,
and CLI diagnostics/audit/support commands using AXIOM's current Control Center
components and CSS. It does not execute repairs automatically or alter CSS.
A successful local unit test run does not replace the full repository test
suite, browser E2E tests, or platform package checks. Roadmap status should only
be updated after those gates have been run against the repository checkout.

## Module Explorer

The Control Center Module Explorer is a searchable index of the existing GUI pages, CLI command families, and the live `/api/docs` OpenAPI reference. GUI entries navigate to the existing page. CLI-only entries show and copy a command rather than attempting to execute a shell command in the browser. Training and workflow entries keep their planning/design-only limitations visible. The explorer is a discoverability layer, not evidence that every planned roadmap feature is shipped.

The Module Explorer only routes into pages that exist in the current GUI. CLI-only families are displayed as CLI-only and use copyable command examples; they are not misrepresented as in-browser command execution. Planned work remains planned until its source roadmap status and release checks support a change.


## Security posture audit

The Diagnostics page, `GET /api/security/audit`, and `axiom security audit`
share the same read-only security findings. The checks cover project-config
permissions and symlink paths, `.axiom/tool_permissions.json` permissions,
active MCP tool grants, API read/write token configuration, the optional
OpenAI-compatible bearer token, and exact CORS origin safety.

Use:

```bash
axiom security audit
axiom security audit --json
axiom security audit --strict --json
axiom security audit --daemon-host 0.0.0.0 --allow-network --json
```

The default daemon profile is loopback-only. The final example evaluates a
proposed non-loopback daemon profile and reports it as unsafe because the
lightweight daemon exposes unauthenticated HTTP endpoints; it does not start or
change a listener. Credential values are never returned. The API reports
whether scopes are configured, not the tokens themselves.

This audit does not scan operating-system listening sockets, inspect host
firewall rules, or infer the bind address of servers started outside AXIOM's
built-in launchers. CORS is not authentication; keep remote origins restricted
to trusted HTTPS applications and configure API token scopes before exposing
the FastAPI API beyond loopback.


## Dependency and release-artifact checks

`axiom doctor` checks that the required runtime packages are importable and
that installed versions meet the minimum versions declared by
`pyproject.toml`. JSON diagnostics distinguish missing distributions from
outdated versions. This is a compatibility and local-install check, not a
vulnerability-database lookup.

The read-only security posture audit additionally checks each required runtime
distribution's installed metadata and verifies hashed package files against
its local `RECORD` manifest. Package inspection is bounded to 512 RECORD entries, 64 hashed files,
or 2 MiB per distribution; install-scheme files outside the package directory
(such as generated console scripts) are not included in that hash pass. A missing or invalid RECORD inventory, a skipped external install-scheme
path, or a reached inspection limit is reported as incomplete rather than
silently treated as verified.


Inspect a locally built wheel or source archive without extracting or running
it:

```bash
axiom security artifact inspect ./dist/axiom_all-0.3.0-py3-none-any.whl
axiom security artifact inspect ./dist/axiom_all-0.3.0-py3-none-any.whl --json
axiom security artifact inspect ./dist/axiom-all-0.3.0.tar.gz --strict --json
```

For wheels, the inspector checks required Core Metadata/WHEEL files, filename
consistency, path safety, and the hashes and sizes recorded in `RECORD`.
For source archives, it checks member paths, duplicate names, link/special-file
entries, build-manifest presence, and archive expansion bounds. It never
extracts files or executes `setup.py`, build backends, or project code.

Inspection is bounded to artifacts up to 256 MiB, 10,000 entries, 256 MiB per
member, and 1 GiB total uncompressed content. Extreme ZIP compression ratios
and encrypted ZIP members fail closed. Signature sidecars are detected but
their publisher trust chain is not verified. A successful structural check is
not a malware scan, a CVE audit, or proof that the artifact came from a trusted
publisher; use a trusted build pipeline, provenance/signature verification,
and a vulnerability database scanner for those controls.
