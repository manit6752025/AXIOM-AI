"""CLI commands for auditing roadmap status and scope integrity."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from axiom.core.roadmap import audit_roadmap

console = Console()
roadmap_app = typer.Typer(help="Inspect the 120-item AXIOM product roadmap.", no_args_is_help=True)


@roadmap_app.command("audit")
def audit(
    path: Path | None = typer.Option(None, "--path", help="Roadmap file; defaults to the source checkout or current workspace."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable report."),
) -> None:
    """Validate roadmap IDs and summarize implemented, in-progress, and planned items."""
    try:
        report = audit_roadmap(path)
    except (OSError, ValueError) as exc:
        message = {"schema_version": 1, "status": "error", "message": str(exc)}
        if json_output:
            typer.echo(json.dumps(message, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from None

    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        if not report["integrity"]["ok"]:
            raise typer.Exit(code=1)
        return

    summary = report["counts"]
    table = Table(title="AXIOM ROADMAP AUDIT")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    table.add_row("Implemented; tests passing", str(summary["implemented"]))
    table.add_row("In progress / validation pending", str(summary["in_progress"]))
    table.add_row("Planned", str(summary["planned"]))
    console.print(table)
    console.print(f"Scope integrity: {'PASS' if report['integrity']['ok'] else 'FAIL'} · {report['total']}/120 items parsed")
    if not report["integrity"]["ok"]:
        console.print_json(json.dumps(report["integrity"], indent=2))
        raise typer.Exit(code=1)


app = roadmap_app
