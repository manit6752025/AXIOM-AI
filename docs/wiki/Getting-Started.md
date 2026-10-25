# Getting Started

This guide covers installing AXIOM v0.3.0-beta.6, validating a workspace, and opening the local Control Center.

## 1. Install

### Python / PyPI

    python -m pip install --upgrade --pre axiom-all
    axiom version

The distribution is named `axiom-all`; the installed command is `axiom`. A virtual environment is recommended.

### Debian / Ubuntu / Linux Mint (amd64)

    curl -fsSLo /tmp/axiom-archive-keyring.gpg https://netcore-technologies.github.io/AXIOM-AI/apt/axiom-archive-keyring.gpg
    sudo install -Dm0644 /tmp/axiom-archive-keyring.gpg /usr/share/keyrings/axiom-archive-keyring.gpg
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/axiom-archive-keyring.gpg] https://netcore-technologies.github.io/AXIOM-AI/apt stable main' | sudo tee /etc/apt/sources.list.d/axiom.list
    sudo apt update
    sudo apt install axiom-ai

The APT package currently targets 64-bit Intel/AMD (`amd64`).

### macOS and Linux installer

    curl -fsSL https://raw.githubusercontent.com/NetCore-Technologies/AXIOM-AI/main/installers/install.sh | bash
    axiom version

The user-scoped installer places `axiom` in `~/.local/bin`. Ensure that directory is on `PATH`.

### Windows PowerShell

    irm https://raw.githubusercontent.com/NetCore-Technologies/AXIOM-AI/main/installers/install.ps1 | iex
    axiom version

Follow the installer output and reopen PowerShell if the `PATH` update is not visible.

> Check the [beta.6 release page](https://github.com/NetCore-Technologies/AXIOM-AI/releases/tag/v0.3.0-beta.6) and [Actions workflow](https://github.com/NetCore-Technologies/AXIOM-AI/actions/workflows/release-assets.yml) before relying on any specific platform binary.

## 2. Learn the command path

    axiom --help
    axiom guide
    axiom summary
    axiom status
    axiom doctor --json

These commands provide orientation and read-only status/diagnostic output.

## 3. Create a workspace

    axiom init my-ai
    cd my-ai
    axiom project validate
    axiom config validate
    axiom system info

Project configuration lives in `axiom.yaml`; the local model registry lives in `.axiom`.

## 4. Open the Control Center

    axiom gui

Use `axiom gui --no-open` to print the local URL without opening a browser automatically. The service is intended for local use and binds to loopback by default.

## 5. Go deeper

- [CLI Guide](CLI-Guide)
- [Control Center and Module Explorer](Control-Center-and-Module-Explorer)
- [Workspace Diagnostics and Support Bundles](Workspace-Diagnostics-and-Support-Bundles)
- [Troubleshooting](Troubleshooting)
