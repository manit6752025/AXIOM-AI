# Cross-platform Python package verification

AXIOM currently publishes a platform-independent Python wheel and a source distribution. The same build contract is exercised on Ubuntu (`ubuntu-latest`), Windows (`windows-latest`), and macOS (`macos-latest`) by the `cross-platform-smoke` job in `.github/workflows/quality-gates.yml`.

For each operating system the job:

1. Installs Node.js 22 dependencies and runs the production UI build, including atomic activation of the bundled Control Center under `axiom/webui`.
2. Builds exactly one wheel and one gzip-compressed source distribution with `python -m build --sdist --wheel`.
3. Runs `scripts/verify_package_artifacts.py` over both archives without extracting them.
4. Installs the wheel with `--no-deps` and imports the installed `axiom` package.

The verifier checks archive member paths for traversal/absolute/drive paths, duplicate and Windows case-colliding names, links and special files, member-count and expanded-size bounds, ZIP CRC integrity, PEP 440 project version and Python requirement metadata, the `axiom = axiom.cli.main:app` console entry point, and the packaged Control Center HTML/JavaScript/CSS. The sdist must contain its root `pyproject.toml`, README and package payload. No archive member is extracted or executed during inspection.

Run the same check locally after building the UI:

```sh
npm ci
npm run build
python -m pip install --upgrade build
python -m build --sdist --wheel --outdir dist
python scripts/verify_package_artifacts.py --dist dist --install-wheel --json
```

The wheel import check is intentionally performed without dependencies: it verifies package installation and the top-level import only. The regular full-validation job separately installs the declared dependencies and exercises the API, CLI, UI and native-inference test suites. Optional hardware drivers and accelerator backends are not certified by package-format verification.
