# Security and Privacy

Treat local datasets, model files, configuration and tokens as sensitive.

Dataset field-name warnings are heuristic signals; they are not proof that
a dataset is safe or unsafe. Review data according to your privacy requirements.

Filesystem access should reject traversal, unsafe symlinks and paths outside
the intended workspace.

Report vulnerabilities using the repository security policy:
https://github.com/manit6752025/AXIOM-AI/blob/main/SECURITY.md


## Software bill of materials and dependency advisories

Diagnostics can export an SPDX 2.3 SBOM for the active Python environment.
It reads installed distribution metadata only and does not execute package
code, collect host paths or contact a network service. The inventory contains
package names, versions, Python package URLs and dependency relationships
where both installed distributions are present. Unavailable license metadata
is represented as `NOASSERTION`; AXIOM does not guess license data.

The separate dependency vulnerability scan invokes `pip-audit --local`
against the installed Python environment only after the operator selects the
action and confirms network authorization for network access. It may contact external advisory services.
AXIOM restricts the subprocess environment so arbitrary API tokens and private
package-index URLs are not passed to the scanner, applies a hard timeout and
output-size bound, and returns structured advisory IDs, aliases and available
fix versions. No package upgrades are performed. If the scanner is not
installed, times out or cannot complete, the report is marked unavailable or
error; that outcome is never described as a clean scan.

Install the optional scanner with `python -m pip install "axiom-all[security-audit]"`; this extra provides `pip-audit` while keeping the scanner out of AXIOM's default local-first runtime dependencies.

CLI equivalents are `axiom security sbom --output axiom.spdx.json` and
`axiom security vulnerabilities --allow-network --json`. Export files are
created without overwriting existing files and use restrictive local
permissions. Review SBOM and advisory reports before sharing them.
