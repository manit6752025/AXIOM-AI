"""Shared safety rules for interactive and scripted CLI execution."""
from __future__ import annotations

import json

import click
import typer


def _root_options() -> dict[str, object]:
    context = click.get_current_context(silent=True)
    if context is None:
        return {}
    root = context.find_root()
    return root.obj if isinstance(root.obj, dict) else {}


def is_interactive() -> bool:
    """Return true only when input is a TTY and no-input mode is disabled."""
    if bool(_root_options().get("non_interactive")):
        return False
    try:
        stream = click.get_text_stream("stdin")
        return bool(stream.isatty())
    except (AttributeError, OSError, RuntimeError):
        return False


def _prompt_error(message: str, hint: str) -> None:
    detail = f"{message} Interactive input is unavailable. {hint}"
    options = _root_options()
    if bool(options.get("json_errors")):
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "error": {
                        "code": "interactive_input_required",
                        "message": detail,
                        "exit_code": 1,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            err=True,
        )
    else:
        typer.echo(f"Error: {detail}", err=True)
    raise typer.Exit(code=1)


def prompt(
    text: str,
    *args: object,
    non_interactive_hint: str = "Supply the corresponding command-line option or an environment variable.",
    **kwargs: object,
) -> str:
    """Prompt only in a real terminal; otherwise fail with actionable guidance."""
    if not is_interactive():
        _prompt_error(text, non_interactive_hint)
    return typer.prompt(text, *args, **kwargs)


def confirm(text: str, *args: object, default: bool = False, **kwargs: object) -> bool:
    """Use the safe default instead of prompting when input is unavailable."""
    if not is_interactive():
        return default
    return typer.confirm(text, *args, default=default, **kwargs)
