# AXIOM AI CLI Launcher

**Build AI. Own AI.**

`@netcore-technologies/axiom-cli` provides a Node.js launcher for the AXIOM
AI command-line interface. This package launches the Python CLI; it is not
the complete AXIOM application.

## Requirements

- Node.js 20 or newer.
- >=3.11
- Python available on PATH.

## Install

    python -m pip install axiom-all
    npm install --global @netcore-technologies/axiom-cli@beta
    axiom --help

## Create a project

    axiom init my-first-ai
    cd my-first-ai

## Beta feature foundations

AXIOM includes a bounded JSONL dataset profiler with row statistics, field
type/null summaries and potential sensitive-field-name warnings. The broader
project includes model inspection, optimisation, quantization, local workspace
APIs and GUI sections for engineering workflows.

Advanced workflow building, persistent evaluation history, hardware telemetry
and additional security diagnostics are part of the roadmap and should not
be assumed available as completed end-to-end features.

## Updating

    npm install --global @netcore-technologies/axiom-cli@beta
    python -m pip install --upgrade axiom-all

## Documentation and support

- [Project README](https://github.com/manit6752025/AXIOM-AI#readme)
- [Roadmap](https://github.com/manit6752025/AXIOM-AI/blob/main/ROADMAP.md)
- [Release notes](https://github.com/manit6752025/AXIOM-AI/blob/main/docs/releases/v0.3.0-beta.1.md)
- [Wiki](https://github.com/manit6752025/AXIOM-AI/wiki)
- [Issues](https://github.com/manit6752025/AXIOM-AI/issues)

The npm launcher license is MIT. Consult the repository license and package
metadata for licensing details of the complete application and dependencies.
