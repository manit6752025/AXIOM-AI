"""Consistent human and machine-readable CLI error output."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import NoReturn

import click
import typer


EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1
EXIT_CODE_USAGE = 2
EXIT_CODE_CANCELLED = 130
_RUNTIME_OPTIONS: dict[str, object] = {}


def set_runtime_options(options: Mapping[str, object]) -> None:
    """Publish the active root output policy for errors raised outside Click callbacks."""
    _RUNTIME_OPTIONS.clear()
    _RUNTIME_OPTIONS.update(options)


def runtime_options() -> dict[str, object]:
    """Read the root command's output policy, with a callback-managed fallback."""
    context = click.get_current_context(silent=True)
    if context is not None:
        root = context.find_root()
        if isinstance(root.obj, dict) and root.obj:
            return dict(root.obj)
    return dict(_RUNTIME_OPTIONS)


def json_errors_enabled(options: Mapping[str, object] | None = None) -> bool:
    """Return whether the current invocation requested JSON error envelopes."""
    selected = options if options is not None else runtime_options()
    return bool(selected.get("json_errors"))


def emit_error(
    message: str,
    *,
    code: str = "AXIOM_CLI_ERROR",
    exit_code: int = EXIT_CODE_FAILURE,
    options: Mapping[str, object] | None = None,
    stderr: bool = False,
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
        typer.echo(f"Error: {message}", err=stderr)
    raise typer.Exit(code=exit_code)
