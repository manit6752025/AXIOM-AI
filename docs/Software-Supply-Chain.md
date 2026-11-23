# Software supply-chain inventory and dependency vulnerability audit

AXIOM exposes two deliberately separate supply-chain operations from
**Diagnostics** and the `axiom security` CLI.

## Export an SPDX 2.3 SBOM

~~~bash
axiom security sbom
axiom security sbom --output ./axiom-installed-environment.spdx.json
~~~

The JSON endpoint is `GET /api/security/sbom`. It inventories installed Python
distribution metadata in the active interpreter. Packages include normalized
PyPI package URLs (PURLs), installed versions, and dependency relationships
only when the referenced dependency is also installed.

The export includes no package file contents, host installation paths,
credentials, or network-derived data. It does not import or execute installed
package code. The content fingerprint is deterministic for the same package
metadata. License fields use `NOASSERTION`: package metadata is not sufficient
to grant or verify redistribution rights.

The export target is create-only. AXIOM refuses to overwrite an existing file,
follow symbolic-link parent components, or write through an output symlink;
new files receive private permissions where supported.

## Explicit dependency vulnerability scan

~~~bash
pip install -e ".[security-audit]"
axiom security vulnerabilities --allow-network --timeout 90 --output ./dependency-audit.json
~~~

The matching endpoint is `POST /api/security/dependency-audit`. A scan does
not run unless the operator explicitly authorizes network access. The
`pip-audit --local` scanner checks the active Python environment against its
online vulnerability advisory sources; it does not install or upgrade
packages.

The subprocess runs with an allowlisted environment, disables pip
configuration files, uses the active interpreter environment as its working
directory rather than the project directory, and has a bounded timeout. Its
stdout is captured up to 8 MiB and stderr up to 1 MiB; the reader continues
draining excess output without retaining it. Oversized output, invalid JSON,
scanner errors, timeouts, or a missing scanner return an explicit
`error`/`unavailable` state and never become a clean bill of health.
Environment values, credentials, package file contents, and network
identifiers are omitted from the report.

The JSON report includes scanner name/version, scan timestamp and scope,
counts, advisory identifiers, aliases, fix versions, and limitations. It
covers installed Python distributions only—not operating-system packages,
browser dependencies, container base images, model runtimes, or undisclosed
software. Review the original advisory and applicability before prioritizing a
finding. A zero-finding result is not a security guarantee.

## Boundaries

SBOM generation is local and metadata-only. The vulnerability audit is a
separate optional network operation; it does not silently run as part of
Diagnostics refresh or SBOM export. Both exports are create-only when written
to disk. Keep generated reports under appropriate access controls and review
them before sharing.
