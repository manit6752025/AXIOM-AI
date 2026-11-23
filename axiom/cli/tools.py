"""CLI commands for discovering and safely setting up developer tools."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import typer

from axiom.cli.errors import emit_error
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.tools.awake import AwakeUnavailableError, KeepAwakeSession
from axiom.tools.catalog import TOOL_CATALOG as TOOLS
from axiom.tools.catalog import InstallCandidate, ToolSpec
from axiom.tools.installer import (
    InstallerSafetyError,
    InstallTarget,
    SecretInputError,
    build_install_plan,
    execute_install_plan,
    update_profile_path,
)

console = Console()


def _platform_name() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def _find_tool(tool_id: str) -> ToolSpec:
    query = tool_id.strip().lower()
    for tool in TOOLS:
        aliases = {tool.id.lower(), tool.name.lower(), *tool.executable_names}
        if query in aliases:
            return tool
    known = ", ".join(tool.id for tool in TOOLS)
    raise ValueError(f"Unknown tool '{tool_id}'. Choose one of: {known}.")


def _current_candidates(tool: ToolSpec) -> tuple[InstallCandidate, ...]:
    current = _platform_name()
    return tuple(
        candidate
        for candidate in tool.install_candidates
        if candidate.platform == current
    )


def _status(tool: ToolSpec) -> str:
    commands = [name for name in tool.executable_names if shutil.which(name)]
    if commands:
        return f"installed ({', '.join(commands)})"
    if tool.api_key_env_vars and any(os.getenv(name) for name in tool.api_key_env_vars):
        return "credential reference present"
    if tool.kind == "api":
        return "API setup needed"
    return "not found"


def _candidate_command(candidate: InstallCandidate) -> str:
    return candidate.command


def _package_candidate(
    tool: ToolSpec,
    *,
    manager: str | None = None,
) -> tuple[str, str, InstallCandidate] | None:
    """Return a package-manager candidate the safe installer understands."""

    allowed = {"npm", "brew", "winget", "choco"}
    requested = manager.strip().lower() if manager else None
    for candidate in _current_candidates(tool):
        parts = shlex.split(candidate.command, posix=os.name != "nt")
        if not parts:
            continue
        command = parts[0].lower()
        if command not in allowed or (requested and command != requested):
            continue

        package: str | None = None
        if command == "npm" and len(parts) == 4 and parts[1:3] == ["install", "-g"]:
            package = parts[3]
        elif (
            command == "brew"
            and len(parts) == 3
            and parts[1] == "install"
            or command == "winget"
            and len(parts) == 3
            and parts[1] == "install"
            or command == "choco"
            and len(parts) == 3
            and parts[1] == "install"
        ):
            package = parts[2]

        if package:
            return command, package, candidate
    return None


def _profile_path() -> Path:
    home = Path.home()
    if os.name == "nt":
        candidates = (
            home / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1",
            home
            / "Documents"
            / "WindowsPowerShell"
            / "Microsoft.PowerShell_profile.ps1",
        )
    else:
        shell = Path(os.environ.get("SHELL", "")).name
        preferred = (
            ".zshrc" if shell == "zsh" else ".bashrc" if shell == "bash" else ".profile"
        )
        candidates = (
            home / preferred,
            home / ".profile",
            home / ".zshrc",
            home / ".bashrc",
        )

    return next((path for path in candidates if path.exists()), candidates[0])


def _command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    return result.stdout.strip()


def _global_bin_path(manager: str) -> Path | None:
    if manager == "npm":
        prefix = _command_output(["npm", "prefix", "-g"])
        if prefix:
            return Path(prefix) if os.name == "nt" else Path(prefix) / "bin"
    if manager == "brew":
        prefix = _command_output(["brew", "--prefix"])
        if prefix:
            return Path(prefix) / "bin"
    return None


def _add_install_path(manager: str) -> str:
    path = _global_bin_path(manager)
    if path is None:
        return "No user PATH directory was inferred; restart the terminal and use the vendor's PATH guidance."

    profile = _profile_path()
    change = update_profile_path(profile, [str(path)], dry_run=False)
    if not change.changed:
        return f"User PATH already includes {path}."
    return f"Added {path} to {profile}; open a new terminal to load it."


def _print_install_plan(
    tool: ToolSpec,
    candidate: InstallCandidate | None,
    plan: object | None,
) -> None:
    lines = [f"Tool: {tool.name}", f"Platform: {_platform_name()}"]
    if candidate is not None:
        lines.extend(
            (
                f"Candidate: {candidate.install_kind}",
                f"Command: {candidate.command}",
                f"Source: {candidate.source_url}",
            )
        )
    if plan is not None:
        plan_dict = plan.as_dict()  # type: ignore[union-attr]
        commands = plan_dict.get("commands", [])
        lines.append(
            "AXIOM command: "
            + (
                " ".join(commands[0])
                if commands
                else "no supported package manager found"
            )
        )
        lines.extend(plan_dict.get("warnings", []))
    console.print(Panel.fit("\n".join(lines), title="AXIOM TOOL SETUP"))


def _manual_setup(tool: ToolSpec) -> None:
    candidates = _current_candidates(tool)
    if not candidates:
        console.print(
            f"[yellow]No {_platform_name()} candidate is cataloged for {tool.name}.[/yellow]"
        )
        return
    console.print(
        f"[yellow]AXIOM will not execute a remote installer script or SDK install automatically.[/yellow]\n"
        f"Review the vendor command for [cyan]{tool.name}[/cyan]:"
    )
    for candidate in candidates:
        console.print(f"  {candidate.command}")
        console.print(f"  Source: {candidate.source_url}")


def register(app: typer.Typer, tools_app: typer.Typer) -> None:
    @tools_app.command("list")
    def list_tools(
        as_json: bool = typer.Option(
            False, "--json", help="Print machine-readable tool metadata."
        ),
    ) -> None:
        """List supported developer tools and their local status."""

        if as_json:
            records = [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "kind": tool.kind,
                    "status": _status(tool),
                    "executables": list(tool.executable_names),
                }
                for tool in TOOLS
            ]
            typer.echo(json.dumps({
                "schema_version": 1,
                "platform": _platform_name(),
                "total": len(records),
                "tools": records,
                "scope_note": "Local catalog and executable/environment-variable presence only; credential values and vendor authentication state are not returned or verified.",
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return

        table = Table(title=f"AXIOM Developer Tools · {_platform_name()}")
        table.add_column("ID")
        table.add_column("Tool")
        table.add_column("Surface")
        table.add_column("Status")
        for tool in TOOLS:
            table.add_row(tool.id, tool.name, tool.kind, _status(tool))
        console.print(table)

    @tools_app.command("doctor")
    def tools_doctor(
        json_output: bool = typer.Option(False, "--json", help="Emit a versioned local tool inventory without exposing credential values."),
    ) -> None:
        """Check executable presence without claiming vendor authentication."""
        if json_output:
            records = []
            for tool in TOOLS:
                installed = [name for name in tool.executable_names if shutil.which(name)]
                credential_present = bool(tool.api_key_env_vars and any(os.getenv(name) for name in tool.api_key_env_vars))
                records.append({
                    "id": tool.id,
                    "name": tool.name,
                    "kind": tool.kind,
                    "installed_executables": installed,
                    "status": _status(tool),
                    "credential_reference_available": credential_present,
                    "vendor_authentication_verified": False,
                })
            typer.echo(json.dumps({
                "schema_version": 1,
                "platform": _platform_name(),
                "total": len(records),
                "tools": records,
                "scope_note": "Local executable and environment-variable presence only; AXIOM does not read, return, or verify credential values or vendor sign-in state.",
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return

        table = Table(title="AXIOM Tool Doctor")
        table.add_column("Tool")
        table.add_column("Executable")
        table.add_column("Status")
        table.add_column("Credential reference")
        for tool in TOOLS:
            executable = next(
                (name for name in tool.executable_names if shutil.which(name)), "-"
            )
            credential = (
                ", ".join(name for name in tool.api_key_env_vars if os.getenv(name))
                or "not checked"
            )
            table.add_row(tool.name, executable, _status(tool), credential)
        console.print(table)
        console.print(
            "[dim]Presence is not authentication; finish sign-in with each vendor.[/dim]"
        )

    @tools_app.command("plan")
    def plan_tool(
        tool_id: str = typer.Argument(
            ..., help="Catalog ID, such as claude-code or opencode."
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit a versioned install-candidate plan without executing commands."),
    ) -> None:
        """Show the current-platform install options without running them."""

        try:
            tool = _find_tool(tool_id)
        except ValueError as exc:
            emit_error(str(exc), code="tool_not_found")

        candidates = _current_candidates(tool)
        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "tool": {"id": tool.id, "name": tool.name, "kind": tool.kind},
                "platform": _platform_name(),
                "purpose": tool.purpose,
                "authentication_guidance": tool.auth_notes,
                "source_url": tool.source_url,
                "candidates": [{
                    "install_kind": candidate.install_kind,
                    "command": candidate.command,
                    "source_url": candidate.source_url,
                    "notes": candidate.notes,
                    "executed": False,
                } for candidate in candidates],
                "scope_note": "This is a catalog preview only. AXIOM did not execute an installer, change PATH, download software, or authenticate with the vendor.",
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return

        console.print(Panel.fit(tool.purpose, title=f"{tool.name} · {tool.id}"))
        console.print(f"Auth: {tool.auth_notes}")
        console.print(f"Source: {tool.source_url}")
        if not candidates:
            console.print(
                f"[yellow]No {_platform_name()} install candidate is cataloged.[/yellow]"
            )
            return
        for candidate in candidates:
            console.print(
                f"\n[cyan]{candidate.install_kind}[/cyan]  {candidate.command}"
            )
            console.print(
                f"  {candidate.notes or 'Review the vendor documentation before running it.'}"
            )
            console.print(f"  {candidate.source_url}")

    @tools_app.command("install")
    def install_tool(
        tool_id: str | None = typer.Argument(None, help="Catalog ID to install."),
        all_tools: bool = typer.Option(
            False,
            "--all",
            help="Process every cataloged tool with a safe package candidate.",
        ),
        yes: bool = typer.Option(
            False, "--yes", help="Execute the reviewed package-manager command."
        ),
        manager: str | None = typer.Option(
            None, "--manager", help="Force npm, brew, winget, or choco."
        ),
        add_to_path: bool = typer.Option(
            True,
            "--add-to-path/--no-add-to-path",
            help="Add a detected user-level bin directory to PATH after install.",
        ),
    ) -> None:
        """Preview tool setup; use --yes for explicit package installation."""

        if all_tools and tool_id:
            emit_error("Choose a tool ID or --all, not both.", code="invalid_tool_selection")
        if not all_tools and not tool_id:
            emit_error("Provide a tool ID or use --all.", code="missing_tool_selection")

        selected = list(TOOLS) if all_tools else [_find_tool(tool_id or "")]
        failures = 0
        for tool in selected:
            package_candidate = _package_candidate(tool, manager=manager)
            if package_candidate is None:
                _manual_setup(tool)
                if yes:
                    console.print(
                        "[dim]Skipped: no safe package-manager candidate for automatic execution.[/dim]"
                    )
                continue

            selected_manager, package, candidate = package_candidate
            try:
                plan = build_install_plan(
                    InstallTarget(
                        name=tool.name,
                        package=package,
                        package_manager=selected_manager,
                        auth_instructions=(tool.auth_notes,),
                    ),
                    system=_platform_name(),
                    package_manager=selected_manager,
                    dry_run=not yes,
                )
                _print_install_plan(tool, candidate, plan)
                if not yes:
                    continue
                execute_install_plan(plan)
                console.print(f"[green]✓ Installed {tool.name}.[/green]")
                if add_to_path:
                    try:
                        console.print(_add_install_path(selected_manager))
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        console.print(
                            f"[yellow]Installed, but PATH was not updated:[/yellow] {exc}"
                        )
            except (InstallerSafetyError, SecretInputError, OSError, ValueError) as exc:
                failures += 1
                console.print(f"[red]Could not set up {tool.name}:[/red] {exc}")

        if failures:
            raise typer.Exit(code=1)

    @app.command("session")
    def session(
        keep_awake: bool = typer.Option(
            False, "--keep-awake", help="Keep the machine awake for the timed session."
        ),
        minutes: float = typer.Option(
            60.0, "--minutes", min=0.1, max=24 * 60, help="Session duration in minutes."
        ),
        reason: str = typer.Option(
            "AXIOM developer session",
            "--reason",
            help="Short reason shown to the OS power helper.",
        ),
    ) -> None:
        """Run a bounded local session, optionally holding a keep-awake lock."""

        if not keep_awake:
            console.print(
                "Use [cyan]axiom session --keep-awake --minutes 60[/cyan] to start a bounded keep-awake session."
            )
            return

        duration = minutes * 60
        try:
            session_handle = KeepAwakeSession(duration=duration, reason=reason)
            if not session_handle.available:
                console.print(
                    "[yellow]Keep-awake is unavailable on this host.[/yellow]"
                )
                for instruction in session_handle.plan().instructions:
                    console.print(f"  {instruction}")
                raise typer.Exit(code=1)
            with session_handle:
                console.print(
                    f"[green]AXIOM session active for {minutes:g} minutes[/green] "
                    f"via {session_handle.plan().executable or 'system helper'}."
                )
                time.sleep(session_handle.duration)
        except AwakeUnavailableError as exc:
            console.print(f"[red]Keep-awake failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
