# Distribution channels

| Channel | Gate |
|---|---|
| GitHub Releases | Existing workflow; validate assets and checksums |
| PyPI | Preserve explicit wheel/sdist checks and `python-dist/` |
| npm | Launcher package and trusted publishing configuration |
| APT | .deb build, Packages/Release metadata, protected GPG key, Pages |
| Homebrew | Tap repo, verified URL and SHA256 |
| Scoop/WinGet | Verified Windows asset/hash; WinGet review may apply |
| Arch AUR | Verified PKGBUILD and AUR account |
| GHCR | Container build and smoke tests |

Never commit signing keys or use placeholder checksums in a published package.
