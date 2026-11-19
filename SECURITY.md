# Security Policy

## Supported Versions

AXIOM is currently under active development.

Security fixes will generally target the latest version on the `main` branch.

| Version        | Supported |
| -------------- | --------- |
| `main`         | ✅         |
| Older releases | ❌         |

---

## Reporting a Vulnerability

Please **do not open a public GitHub issue for a security vulnerability**.

Instead, report security issues privately through the repository's GitHub security reporting mechanism.

When reporting a vulnerability, please include:

* a clear description of the issue
* affected component or feature
* steps to reproduce
* potential impact
* relevant logs or proof of concept
* any suggested mitigation

Please avoid including secrets, credentials, private datasets, API keys, or other sensitive information in the report.

---

## What to Report

Examples of security issues include:

* remote code execution
* arbitrary command execution
* authentication bypass
* privilege escalation
* unsafe model loading
* malicious model or dataset execution
* path traversal
* insecure API endpoints
* accidental exposure of credentials
* unauthorized access to private data
* vulnerabilities in AXIOM's deployment mechanisms

---

## Model and Dataset Security

AI systems introduce risks beyond traditional software.

AXIOM may eventually process:

* downloaded models
* user-provided datasets
* generated files
* external repositories
* inference inputs
* training artifacts

Treat third-party models and datasets as **untrusted inputs** unless their origin and integrity are known.

Do not run untrusted model files or training code with unnecessary system privileges.

---

## Disclosure

We aim to investigate valid security reports promptly and coordinate responsible disclosure where appropriate.

Once an issue has been fixed, relevant security information may be published through the repository's release notes or security advisories.

---

## Scope

This policy applies to the AXIOM open-source project and its official repositories.

Third-party integrations, hosted infrastructure, and external services may have separate security policies.

## Beta.5 security hardening

Beta.5 fixes the reported model-path injection risks and administrator clear-text browser-storage issue. Report new vulnerabilities privately through GitHub security reporting. The public release record is in `docs/releases/v0.2.0-beta.5.md`.

<!-- AXIOM-BETA5-RELEASE -->

## Beta.5 security fixes

Beta.5 hardens model-path handling against path traversal/path injection and removes administrator verifier material from clear-text browser storage. Report security problems through the repository security policy rather than public issue disclosure when sensitive exploitation details are involved.

## v0.2.0-beta.5 security update

Beta.5 includes hardening for untrusted model path handling and administrator browser storage.

Security issues should be reported privately through GitHub's repository security process.

## Runtime bundle path advisory — 2026-10-08

The CodeQL path-expression findings reported in the model policy, optimizer,
and model-path surfaces were reviewed on 2026-10-08. The confirmed runtime
bundle issues were source-symlink disclosure and destination-symlink
redirection in `create_runtime_bundle`. They were patched on 2026-10-08 by
validating every resolved source and destination child, rejecting symlinks,
and keeping runtime-profile writes below the trusted output root.

See the complete incident timeline, impact, patch, and regression coverage in
[SECURITY-ADVISORY-2026-10.md](docs/SECURITY-ADVISORY-2026-10.md).

## Beta.5 security update

Beta.5 hardens administrator browser storage and untrusted model filesystem path handling.

## v0.2.0-beta.7 security update

This release includes model filesystem-boundary and administrator-storage hardening. Report suspected vulnerabilities privately; do not publish exploit details in public issues.

## Beta.7 security status

Beta.7 is a pre-release. Open CodeQL Python path-injection alerts remain under investigation; this release does not claim those findings have been fixed. Please report new vulnerabilities privately through GitHub's Security Advisories feature.

## macOS beta.5 distribution images

The beta.5 DMGs are currently unsigned and not notarized. Download only from the official GitHub release, verify the accompanying SHA-256 checksum, and inspect the installer script before running it. Do not treat an open CodeQL alert as resolved until GitHub reports it closed after analysis of the fixed commit.

## Beta 0.2.0-beta.7 status

The beta.7 release includes cross-platform packaging and validation updates. Open CodeQL path-injection alerts are tracked as unresolved until a subsequent scan closes them. Do not rely on this beta as confirmation that every path-injection finding has been remediated. Report suspected vulnerabilities privately through GitHub Security Advisories.
