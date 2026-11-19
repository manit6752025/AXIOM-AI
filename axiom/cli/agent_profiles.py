"""CLI commands for versioned task metadata profiles."""
from __future__ import annotations

import json
import typer

from axiom.cli.errors import emit_error
from rich.console import Console
from rich.table import Table

from axiom.core.agent_profiles import AgentProfileError, AgentProfileStore, TASK_CAPABILITIES


def register(app: typer.Typer) -> None:
    command = typer.Typer(help="Manage descriptive agent task profiles; does not grant tool permissions.", no_args_is_help=True)
    app.add_typer(command, name="agent-profile")
    console = Console()

    @command.command("list")
    def list_profiles(json_output: bool = typer.Option(False, "--json", help="Emit machine-readable profile metadata.")) -> None:
        """List the workspace-local agent profiles."""
        try:
            document = AgentProfileStore().load()
        except AgentProfileError as exc:
            emit_error(str(exc), code="agent_profile_error")
        if json_output:
            typer.echo(json.dumps(document, indent=2, sort_keys=True))
            return
        table = Table(title="AXIOM AGENT PROFILES")
        table.add_column("Name")
        table.add_column("Version", justify="right")
        table.add_column("Task capabilities")
        for profile in document["profiles"]:
            table.add_row(profile["name"], str(profile["version"]), ", ".join(profile["task_capabilities"]) or "none (default-deny)")
        console.print(table)
        console.print("Task profiles do not grant MCP/tool permissions; those remain governed by the MCP policy.")

    @command.command("create")
    def create_profile(
        name: str = typer.Argument(..., help="Display name, 1–64 safe characters"),
        capability: list[str] = typer.Option([], "--capability", help=f"Repeatable task capability: {', '.join(TASK_CAPABILITIES)}"),
        json_output: bool = typer.Option(False, "--json", help="Emit the created profile as JSON."),
    ) -> None:
        """Create a new default-deny descriptive profile."""
        try:
            profile = AgentProfileStore().create(name, capability)
        except AgentProfileError as exc:
            emit_error(str(exc), code="agent_profile_error")
        if json_output:
            typer.echo(json.dumps(profile, indent=2, sort_keys=True))
        else:
            console.print(f"[green]Created profile[/green] {profile['name']} ({profile['id']}); version {profile['version']}.")

    @command.command("set-capabilities")
    def set_capabilities(
        profile_id: str = typer.Argument(..., help="Profile ID, not the display name"),
        capability: list[str] = typer.Option([], "--capability", help="Repeat for each desired capability; no options means clear all."),
        json_output: bool = typer.Option(False, "--json", help="Emit the updated profile as JSON."),
    ) -> None:
        """Replace the descriptive task capability list and increment profile version."""
        try:
            profile = AgentProfileStore().update_capabilities(profile_id, capability)
        except AgentProfileError as exc:
            emit_error(str(exc), code="agent_profile_error")
        if json_output:
            typer.echo(json.dumps(profile, indent=2, sort_keys=True))
        else:
            console.print(f"[green]Updated[/green] {profile['name']} to version {profile['version']}.")

    @command.command("delete")
    def delete_profile(
        profile_id: str = typer.Argument(..., help="Profile ID"),
        yes: bool = typer.Option(False, "--yes", help="Confirm deletion non-interactively."),
    ) -> None:
        """Delete a non-default profile; requires explicit confirmation."""
        if not yes:
            console.print("Refusing to delete without --yes.")
            raise typer.Exit(code=2)
        try:
            result = AgentProfileStore().delete(profile_id)
        except AgentProfileError as exc:
            emit_error(str(exc), code="agent_profile_error")
        console.print(f"[green]Deleted[/green] {result['deleted_id']}.")
