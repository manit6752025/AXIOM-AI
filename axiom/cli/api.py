"""CLI commands for inspecting AXIOM's published API contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from axiom.core.api_examples import ApiExamplesError, build_api_examples, render_api_examples_markdown

api_app = typer.Typer(
    help="Inspect the typed API contract and generate safe request/response examples.",
    no_args_is_help=True,
)
console = Console()


@api_app.command("examples")
def api_examples(
    format_: Literal["markdown", "json"] = typer.Option(
        "markdown", "--format", help="Output format: markdown or json.",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Create a new output file; existing files are never overwritten.",
    ),
) -> None:
    """Generate typed examples from the live OpenAPI contract without making API requests."""
    try:
        from axiom.api.server import create_app

        report = build_api_examples(create_app().openapi())
        rendered = (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            if format_ == "json"
            else render_api_examples_markdown(report)
        )
        if output is not None:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
            console.print(f"Created {format_} API contract examples: {output}")
            return
        typer.echo(rendered, nl=not rendered.endswith("\n"))
    except FileExistsError:
        raise typer.BadParameter("The output file already exists; choose a new path.") from None
    except (OSError, ApiExamplesError) as exc:
        raise typer.Exit(code=1) from exc
