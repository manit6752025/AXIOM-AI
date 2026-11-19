# Workspace diagnostics and support bundles

AXIOM's Diagnostics page and CLI use the same read-only workspace checks. The
checks cover workspace discovery, `axiom.yaml` parsing, unsafe symlink entries,
filesystem read/write access, free disk space, Python compatibility, and the
required Python package imports. Results are structured with stable IDs so the
GUI, scripts, and tests can interpret them without parsing human-oriented text.

## Commands

```bash
axiom doctor
axiom doctor --json
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

## Roadmap audit

`axiom roadmap audit` and the Control Center Roadmap page parse the scoped
120-item section from `ROADMAP.md`. The audit reports the source labels without
inferring completion from file existence, validates R001–R120 coverage, and
surfaces duplicate IDs, missing IDs, unexpected IDs, and unknown statuses.
It is read-only and does not change roadmap labels. A roadmap entry should only
be marked implemented after its applicable backend, GUI, CLI, regression tests,
and documentation gates have been verified in a real checkout.

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
