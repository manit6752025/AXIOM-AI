# Troubleshooting

Collect enough detail to reproduce a problem while minimizing exposure of private data.

## Confirm the environment

    axiom version
    python --version
    axiom --help

Check that the intended Python environment and `axiom` executable are active. For user-scoped installs, confirm `~/.local/bin` is on `PATH`.

## Run diagnostics

    axiom doctor
    axiom doctor --json
    axiom status --json

Review workspace discovery, configuration parsing, filesystem access, disk space, Python compatibility, and required imports. Diagnostics are read-only and do not automatically repair a project.

## GUI or API errors

- Start the Control Center from the intended workspace with `axiom gui`.
- Use `axiom gui --no-open` to copy the local URL yourself.
- For daemon issues, run `axiom daemon --host 127.0.0.1 --port 0` and inspect its startup output.
- Record the sanitized request/correlation ID if one is shown.
- Do not expose the daemon beyond loopback unless you have deliberately evaluated the consequences.

## Create a support bundle

    axiom support-bundle --output ./axiom-support-bundle.zip

Open and inspect the ZIP before attaching it. Metadata minimization does not guarantee that every combination of environment details is anonymous.

## File a useful issue

Include AXIOM version, OS and Python version, installation method, exact command or GUI path, expected versus actual behavior, and a minimal safe reproducer. Never include credentials, API keys, passwords, private dataset rows, proprietary model weights, unredacted logs, or an unreviewed support ZIP.

[Open GitHub Issues](https://github.com/manit6752025/AXIOM-AI/issues) · [Security page](https://github.com/manit6752025/AXIOM-AI/security)
