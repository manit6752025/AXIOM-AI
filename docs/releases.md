# AXIOM release builds

## v0.3.0-beta.6 — Beta candidate

The source metadata and installer examples target `v0.3.0-beta.6`. This is a candidate release note, not confirmation that assets have been built, tested, or published. Check the [GitHub Releases page](https://github.com/NetCore-Technologies/AXIOM-AI/releases) and [release workflow](https://github.com/NetCore-Technologies/AXIOM-AI/actions/workflows/release-assets.yml) for actual status.

AXIOM releases package the terminal CLI and its local daemon. The supported
product surface is the static landing page, the Python command line, and the
dependency-free loopback daemon; there is no hosted workspace or browser
control center in this release line.

## Linux

The Linux workflow can publish x86_64:

- `linux-x64.tar.gz` — recommended for repeated terminal use; the PyInstaller
  runtime remains unpacked between launches.
- `.deb` — Debian package using the same onedir runtime.
- `.AppImage` — portable package.
- `.elf` — one-file executable for a single-file portable path.
- `.sh` — checksum-verifying installer companion.
- `SHA256SUMS` — checksums for the published assets.

The shell installer verifies the tarball checksum before activating it. Linux
architectures without a native bundle use the Python fallback described in the
[README](../README.md) and require Python 3.11+.

## Windows

The release workflow can publish Windows x64 executable, MSI, batch, and
PowerShell assets. The PowerShell installer places the executable under
`%LOCALAPPDATA%\AXIOM` and adds that directory to the user `PATH`.

## macOS

The macOS workflow packages separate arm64 and x64 CLI DMGs and tarballs. The
bundles contain the standalone `axiom` command and an installer helper. They
are not Apple Developer ID signed or notarized unless a future workflow adds
that capability.

## PyPI

The Python distribution is named `axiom-all` and installs the `axiom` command:

```bash
python -m pip install --upgrade --pre axiom-all
axiom version
```

The `publish-pypi.yml` workflow builds and validates a wheel and source
distribution for a published GitHub release, then uses PyPI Trusted Publishing
through the repository's `pypi` environment. It does not store a PyPI token in
the repository. The PyPI project owner must register this exact publisher:

- Owner: `NetCore-Technologies`
- Repository: `AXIOM-AI`
- Workflow: `.github/workflows/publish-pypi.yml`
- Environment: `pypi`

Until that one-time PyPI publisher registration is complete, the workflow can
still build and validate distributions but will not upload them.

## Beta.8 status

[Beta.8 notes](releases/v0.2.0-beta.8.md) describe the target and its known
boundaries. The GitHub [Actions runs](https://github.com/NetCore-Technologies/AXIOM-AI/actions)
and [Releases page](https://github.com/NetCore-Technologies/AXIOM-AI/releases)
are authoritative for the exact commit, workflow result, uploaded assets, and
checksums. A changelog entry or a local build is not release evidence by itself.

## Older notes

- [Beta.6 — efficient Linux distribution](releases/v0.2.0-beta.6.md)
- [Beta.7](releases/v0.2.0-beta.7.md)
- [Beta.5](releases/v0.2.0-beta.5.md)
