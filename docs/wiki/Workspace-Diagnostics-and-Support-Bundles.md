# Workspace Diagnostics and Support Bundles

AXIOM's CLI and Control Center share read-only workspace checks intended to identify common setup problems without silently changing a project.

## Commands

    axiom doctor
    axiom doctor --json
    axiom status --json
    axiom roadmap audit
    axiom roadmap audit --json
    axiom support-bundle --output ./axiom-support-bundle.zip
    axiom support-bundle --output ./axiom-support-bundle.zip --force --json

Use `--force` only when intentionally replacing an existing output file. Without it, the CLI refuses to overwrite; output symlinks are rejected.

## Checks

The diagnostics path covers workspace discovery, `axiom.yaml` parsing, unsafe symlink entries, filesystem read/write access, free disk space, Python compatibility, and required package imports. Results include stable IDs so automation can consume them without scraping human prose.

Set `AXIOM_WORKSPACE=/path/to/project` to select a workspace. `--workspace` explicitly overrides it for `doctor` and `support-bundle`. Otherwise, discovery selects the nearest parent containing a regular `axiom.yaml`, then falls back to the current directory. Caller-provided workspace symlinks are rejected.

## Roadmap audit

The audit checks coverage of the scoped R001–R120 roadmap IDs and reports missing, duplicate, unexpected, or unknown IDs/statuses. It reports recorded labels; it does not infer completion from file presence or automatically update statuses.

## Support-bundle privacy

The documented ZIP contains `diagnostics.json`, `environment.json`, and a static `README.txt`. It excludes configuration contents, environment-variable values, credentials, absolute paths, model weights, datasets, logs, experiment records, and user-authored data. Aggregate checks, application/Python versions, OS family, and dependency versions may still identify an environment in combination.

**Inspect the archive before sharing it.** Data minimization is not a guarantee of anonymity. Do not include secrets, private rows, proprietary weights, or unredacted logs in an issue.

Related: [Troubleshooting](Troubleshooting), [Local API and Security](Local-API-and-Security).
