# Installing AXIOM

Choose the installation method that fits your machine. The installer downloads published release assets; it does not build models, install model weights, or configure provider credentials.

## Quick install

### Windows (PowerShell)

Open PowerShell as your normal user; administrator rights are not required. Run:

```powershell
irm https://raw.githubusercontent.com/NetCore-Technologies/AXIOM-AI/main/installers/install.ps1 | iex
```

The script finds the latest published release, downloads the Windows x64 executable and checksum manifest, verifies SHA-256 before installing, adds the per-user directory to PATH, and runs a version check. It asks before replacing an existing installation.

To inspect the script first or pin a release, download it and run it locally:

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/NetCore-Technologies/AXIOM-AI/main/installers/install.ps1 -OutFile .\install-axiom.ps1
Get-Help .\install-axiom.ps1 -Full
.\install-axiom.ps1 -Tag v0.3.0-beta.6
```

Options include `-Tag` to pin a published release, `-InstallDir` to choose the install directory, `-Force` to skip the overwrite prompt, and `-NoPathUpdate` to leave PATH unchanged. The installer refuses a missing or mismatched checksum.

### Linux x86_64, macOS, and other Linux architectures

```bash
curl -fsSL https://raw.githubusercontent.com/NetCore-Technologies/AXIOM-AI/main/installers/install.sh | bash
```

On Linux x86_64, the script downloads the onedir release bundle and verifies it against `SHA256SUMS` before activation. On macOS and non-x86 Linux, it creates a Python virtual environment and installs from the selected source ref; Python 3.11+ and Git are required for these source installs.

## Useful Bash options

Set environment variables before running the script to customise the installation:

| Variable | Purpose |
| --- | --- |
| `AXIOM_RELEASE_TAG` | Pin a published release, e.g. `v0.3.0-beta.6` |
| `AXIOM_SOURCE_REF` | Git ref for source-based installs |
| `AXIOM_INSTALL_DIR` | Absolute path for release files and the virtual environment |
| `AXIOM_BIN_DIR` | Absolute path for the `axiom` launcher |

Use `--help` to see shell installer usage. Both installers verify downloaded release payloads before use.

## Verify and try it

Open a new terminal after installation and run:

```bash
axiom version
axiom --help
axiom doctor
axiom init my-ai
cd my-ai
axiom project validate
axiom config validate
```

On PowerShell, the same commands work after reopening the terminal. If `axiom` is not found, check that `~/.local/bin` (Linux/macOS) or `%LOCALAPPDATA%\AXIOM` (Windows) is on PATH.

## Troubleshooting

- **No published release found:** open [GitHub Releases](https://github.com/NetCore-Technologies/AXIOM-AI/releases). A tag in the source tree is not necessarily a published release with downloadable assets.
- **Checksum missing or incorrect:** stop; do not bypass verification. Check the release assets and retry later.
- **Python or Git missing:** source installs on macOS/non-x86 Linux require Python 3.11+ and Git.
- **Command not found:** reopen the terminal after PATH changes, or add the printed bin directory to PATH.
- **Permission errors:** prefer a user-owned install directory; do not run the installer as root unless you intentionally configured user paths and understand the consequences.

## Privacy and network behavior

The installer contacts GitHub to identify a release and download assets. It does not include API keys or model weights. Local project checks run on your machine; commands that query model hubs or providers may use the network. Review a generated support bundle before sharing it.

See the [project README](../README.md), [CLI guide](../docs/CLI-GUIDE.md), and [changelog](../docs/CHANGELOG.md) for next steps and release details.
