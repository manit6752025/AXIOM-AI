# AXIOM v0.2.0 Beta 1

The first unified AXIOM release focused on local AI engineering from the
terminal.

## Highlights

### AXIOM Platform

- Python AXIOM platform and CLI
- Model registry and inspection
- Hugging Face model analysis
- Model download safety checks
- Dataset inspection and cleaning
- Hardware detection
- Hardware-aware training planning
- MCP tooling
- SuperCompress integration
- Core platform configuration and runtime foundations

### Repository

AXIOM keeps the CLI, local daemon, static landing page, and supporting
engineering modules together in one repository:

https://github.com/manit6752025/AXIOM-AI

## GitHub Pages

The AXIOM landing page is deployed through GitHub Pages.

## Downloads

This release provides platform-specific AXIOM executable builds through the GitHub Release assets.

Available packages include:

- Windows x64 portable executable
- Windows installer script
- Linux x64 executable
- Linux AppImage
- Linux Debian package
- Linux installer script
- SHA256 checksums

## Beta Status

AXIOM is still under active development.

The local daemon exposes a small health/info boundary. Full model serving,
training execution, evaluation execution, and production runtime serving are
still under development.

This beta establishes the unified AXIOM CLI, daemon, and release pipeline.

## Project

**AXIOM — Build AI. Own AI.**

Built by **NetCore Technologies**.

## v0.2.0-beta.5

Beta.5 is the optimizer, quantization, and security-hardening beta release.

# v0.2.0-beta.9 preparation

See [preparation notes](v0.2.0-beta.9.md). The notes do not replace the
published GitHub release, successful workflow runs, or checksum verification.

## macOS DMG assets

For beta releases, the macOS workflow builds separate Apple Silicon (arm64)
and Intel (x86_64) DMGs, packages the standalone AXIOM CLI, generates
SHA-256 checksums, and uploads assets only when the workflow succeeds. These
images are unsigned and not notarized unless a future signing/notarization
workflow is configured.
