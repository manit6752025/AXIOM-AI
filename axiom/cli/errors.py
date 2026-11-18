"""Consistent human and machine-readable CLI error output."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import NoReturn

import click
import typer


def runtime_options() -> dict[str, object]:
    """Read the root command's output policy from the active Click context."""
    context = click.get_current_context(silent=True)
    if context is None:
        return {}
    root = context.find_root()
    return root.obj if isinstance(root.obj, dict) else {}


def json_errors_enabled(options: Mapping[str, object] | None = None) -> bool:
    """Return whether the current invocation requested JSON error envelopes."""
    selected = options if options is not None else runtime_options()
    return bool(selected.get("json_errors"))


def emit_error(
    message: str,
    *,
    code: str = "AXIOM_CLI_ERROR",
    exit_code: int = 1,
    options: Mapping[str, object] | None = None,
) -> NoReturn:
    """Emit one stable error envelope or a concise human-readable error."""
    if json_errors_enabled(options):
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "error": {
                        "code": code,
                        "message": str(message),
                        "exit_code": exit_code,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            err=True,
        )
    else:
        typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=exit_code)
