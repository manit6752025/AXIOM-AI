"""Optional, explicit integration with the external Headroom CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence

import typer
from rich.console import Console
from rich.panel import Panel

HEADROOM_SOURCE_URL = "https://github.com/headroomlabs-ai/headroom"
HEADROOM_COMMANDS = {
    "doctor": "headroom doctor",
    "proxy": "headroom proxy --port 8787",
    "dashboard": "headroom dashboard",
}

console = Console()


def _executable() -> str | None:
    """Return the external CLI path without executing or contacting it."""

    return shutil.which("headroom")


def status_payload() -> dict[str, object]:
    """Build a secret-free, local-only Headroom availability record."""

    executable = _executable()
    return {
        "id": "headroom",
        "name": "Headroom",
        "kind": "optional-local-proxy",
        "available": executable is not None,
        "executable": executable,
        "commands": dict(HEADROOM_COMMANDS),
        "source_url": HEADROOM_SOURCE_URL,
        "axiom_behavior": (
            "AXIOM only checks PATH for the external CLI; it does not install Headroom, "
            "start it during status checks, read credentials, send prompts, or measure savings."
        ),
    }


def _print_status(*, json_output: bool = False) -> None:
    payload = status_payload()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if payload["available"]:
        availability = f"available at {payload['executable']}"
    else:
        availability = "not found on PATH"

    console.print(
        Panel.fit(
            "Headroom is an optional local proxy operated by its own CLI.\n"
            f"Status: {availability}\n\n"
            "AXIOM only checks whether the external command is present. It does not "
            "install Headroom, start it during status checks, read credentials, send "
            "prompts, or measure savings.\n\n"
            "Documented commands:\n"
            "  headroom doctor       Check Headroom's local setup.\n"
            "  headroom proxy        Start the optional local proxy.\n"
            "  headroom dashboard    View the dashboard while the proxy is running.\n\n"
            f"Reference: {HEADROOM_SOURCE_URL}",
            title="AXIOM HEADROOM",
        )
    )


def _run_external(arguments: Sequence[str]) -> None:
    executable = _executable()
    if executable is None:
        console.print(
            "[yellow]Headroom is not installed or is not on PATH.[/yellow]\n"
            f"Install it using the vendor's instructions, then rerun: {HEADROOM_SOURCE_URL}"
        )
        raise typer.Exit(code=1)

    command = [executable, *arguments]
    console.print(
        "[dim]Delegating to the external Headroom CLI. AXIOM does not send data through it.[/dim]"
    )
    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        console.print(f"[red]Could not start Headroom:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if result.returncode:
        raise typer.Exit(code=result.returncode)


def register(app: typer.Typer) -> None:
    """Register the optional Headroom status and explicit command passthroughs."""

    headroom_app = typer.Typer(
        help="Inspect and explicitly run the optional local Headroom proxy.",
        invoke_without_command=True,
    )
    app.add_typer(headroom_app, name="headroom")

    @headroom_app.callback()
    def headroom_callback(
        ctx: typer.Context,
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Print the local availability record as JSON.",
        ),
    ) -> None:
        if ctx.invoked_subcommand is None:
            _print_status(json_output=json_output)

    @headroom_app.command("status")
    def status(
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Print the local availability record as JSON.",
        ),
    ) -> None:
        """Check whether the external Headroom CLI is available on PATH."""

        _print_status(json_output=json_output)

    @headroom_app.command("doctor")
    def doctor() -> None:
        """Run Headroom's documented local health check explicitly."""

        _run_external(("doctor",))

    @headroom_app.command("proxy")
    def proxy(
        port: int = typer.Option(
            8787,
            "--port",
            min=1,
            max=65535,
            help="Port passed to Headroom's documented proxy command.",
        ),
    ) -> None:
        """Start Headroom's documented local proxy explicitly."""

        _run_external(("proxy", "--port", str(port)))

    @headroom_app.command("dashboard")
    def dashboard() -> None:
        """Open Headroom's documented dashboard explicitly."""

        _run_external(("dashboard",))
