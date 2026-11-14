import json
import logging
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import webbrowser
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread

import click
import inspect
import typer
from click.shell_completion import get_completion_class
from typer.main import get_command
from huggingface_hub import HfApi, login, logout, snapshot_download, whoami
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.cli.ai_features import ai_app
from axiom.cli.optimizer import optimizer_app
from axiom.cli.startup import show_startup, suppress_for_arguments
from axiom.cli.tools import register as register_tools_cli
from axiom.cli.headroom import register as register_headroom_cli
from axiom.cli.workflows import register_workflow_cli
from axiom.cli.evaluation import register_evaluation_cli
from axiom.cli.inference import infer_app
from axiom.cli.llm import llm_app
from axiom.cli.mobile_bridge import register_mobile_bridge
from axiom.cli.engine import engine_app
from axiom.core.hardware import detect_hardware, estimate_model_fit
from axiom.core.experiments import ExperimentStore
from axiom.core.integrations import (
    AgentIntegration,
    IntegrationRegistry,
    IntegrationType,
)
from axiom.core.project import create_project
from axiom.core.tool_permissions import TOOL_DESCRIPTIONS, ToolPermissionStore
from axiom.daemon import LOCAL_ACTIONS, run_daemon
from axiom.datasets.cleaner import clean_jsonl
from axiom.datasets.inspector import inspect_dataset
from axiom.models.analysis import analyze_config, disk_info
from axiom.models.inspector import inspect_model
from axiom.models.registry import Model, ModelRegistry
from axiom.core.storage import atomic_write_text
from axiom.runtime.supercompress import compress_context, redacted_base_url
from axiom.training.planner import create_training_plan
from axiom.version import __version__

app = typer.Typer(
    name="axiom",
    help="Inspect AI work, plan around your hardware, and keep it local.",
    invoke_without_command=True,
    add_completion=True,
)

model_app = typer.Typer(help="Manage AI models.")
dataset_app = typer.Typer(help="Inspect and manage datasets.")
system_app = typer.Typer(help="Inspect system hardware and capabilities.")

hf_app = typer.Typer(help="Authenticate and manage Hugging Face access.")
supercompress_app = typer.Typer(help="Use SuperCompress before inference.")
integration_app = typer.Typer(help="Connect AXIOM to AI agents and runtimes.")
mcp_app = typer.Typer(help="Run AXIOM as an MCP server.")
mcp_profile_app = typer.Typer(help="Manage named default-deny MCP tool profiles.", no_args_is_help=True)
tools_app = typer.Typer(help="Discover and safely set up developer tools.")

app.add_typer(model_app, name="model")
app.add_typer(dataset_app, name="dataset")
app.add_typer(system_app, name="system")
app.add_typer(hf_app, name="hf")
app.add_typer(supercompress_app, name="supercompress")
app.add_typer(integration_app, name="integration")
app.add_typer(mcp_app, name="mcp")
mcp_app.add_typer(mcp_profile_app, name="profile")
app.add_typer(tools_app, name="tools")
app.add_typer(llm_app, name="llm")
register_mobile_bridge(app)

app.add_typer(ai_app, name="ai")
app.add_typer(optimizer_app, name="optimize")
train_app = typer.Typer(help="Plan and manage AI training jobs.")
experiment_app = typer.Typer(help="Review saved local experiment plans.", no_args_is_help=True)
workflow_app = typer.Typer(help="Inspect saved local workflow designs.", no_args_is_help=True)
evaluation_app = typer.Typer(help="Run and inspect local offline evaluation reports.", no_args_is_help=True)

app.add_typer(train_app, name="train")
app.add_typer(experiment_app, name="experiment")
app.add_typer(workflow_app, name="workflow")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(infer_app, name="infer")
app.add_typer(engine_app, name="engine")
register_workflow_cli(workflow_app)
register_evaluation_cli(evaluation_app)

# Extended AXIOM CLI
from axiom.cli.dataset_profile import register as register_dataset_profile
from axiom.cli.extended import dataset_inspection_payload, register as register_extended_cli
from axiom.cli.agent_profiles import register as register_agent_profiles
from axiom.cli.model_studio import register_model_studio
from axiom.cli.fine_tuning import register as register_fine_tuning_cli

register_extended_cli(app, model_app, dataset_app)
register_dataset_profile(dataset_app)
register_model_studio(model_app)
register_agent_profiles(app)
register_tools_cli(app, tools_app)
register_headroom_cli(app)
register_fine_tuning_cli(train_app)


console = Console()
_RUNTIME_OPTIONS: dict[str, object] = {}


@app.callback()
def cli_callback(
    ctx: typer.Context,
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Skip the interactive startup banner.",
        is_eager=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show diagnostic logging and detailed operational output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress normal terminal output; errors remain visible."),
    json_errors: bool = typer.Option(False, "--json-errors", help="Emit stable JSON envelopes for AXIOM CLI errors."),
) -> None:
    """Configure global output behavior before running a command."""
    ctx.ensure_object(dict)
    runtime = {"verbose": verbose, "quiet": quiet, "json_errors": json_errors}
    ctx.obj.update(runtime)
    _RUNTIME_OPTIONS.clear()
    _RUNTIME_OPTIONS.update(runtime)
    console.quiet = quiet
    logging.getLogger("axiom").setLevel(logging.DEBUG if verbose else logging.WARNING)

    if verbose and quiet:
        _cli_error("--verbose and --quiet cannot be used together.", code="invalid_global_options")

    if (
        ctx.resilient_parsing
        or ctx.invoked_subcommand in {"mcp", "completion"}
        or suppress_for_arguments(sys.argv[1:])
    ):
        return

    if ctx.invoked_subcommand is None:
        _run_default_session(show_banner=not no_banner and not quiet)
        return

    if no_banner or quiet:
        return

    show_startup(include_next_steps=False)


def _cli_runtime_options() -> dict[str, object]:
    """Return root CLI options for the currently executing command."""
    ctx = click.get_current_context(silent=True)
    if ctx is not None:
        root = ctx.find_root()
        if isinstance(root.obj, dict) and root.obj:
            return root.obj
    return dict(_RUNTIME_OPTIONS)


def _cli_error(message: str, *, code: str = "AXIOM_CLI_ERROR") -> None:
    """Emit a stable human or machine-readable error without leaking details."""
    options = _cli_runtime_options()
    if options.get("json_errors"):
        typer.echo(json.dumps({
            "schema_version": 1,
            "error": {"code": code, "message": message, "exit_code": 1},
        }, ensure_ascii=False, separators=(",", ":")), err=True)
    else:
        typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _run_default_session(*, show_banner: bool = True) -> None:
    """Make bare ``axiom`` the clear, useful entry point for a terminal."""

    if show_banner:
        show_startup(include_next_steps=False)

    console.print(
        Panel.fit(
            "AXIOM inspects models, validates datasets, reads your hardware, "
            "and suggests the next local step.\n"
            "The daemon below exposes the same local context as JSON.",
            title="AXIOM — start here",
        )
    )

    table = Table(title="Useful commands", show_header=False, pad_edge=False)
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("What it does")
    for action in LOCAL_ACTIONS:
        table.add_row(action["command"], action["description"])
    table.add_row(
        "axiom daemon",
        "Run the loopback daemon on a free port and print health/summary URLs.",
    )
    console.print(table)
    console.print(
        "\nLocal daemon: Ctrl-C stops it. Network access stays disabled by default.\n"
    )

    try:
        run_daemon()
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))


def _powershell_completion_script(command: click.Command) -> str:
    """Build a no-install PowerShell completer from AXIOM's actual Click command tree."""
    def describe(current: click.Command) -> dict[str, object]:
        options: set[str] = set()
        for parameter in current.params:
            if isinstance(parameter, click.Option):
                options.update(
                    flag
                    for flag in (*parameter.opts, *parameter.secondary_opts)
                    if flag.startswith("-")
                )
        commands: dict[str, object] = {}
        if isinstance(current, click.Group):
            commands = {
                name: describe(subcommand)
                for name, subcommand in sorted(current.commands.items())
            }
        return {"options": sorted(options), "commands": commands}

    schema = json.dumps(describe(command), ensure_ascii=False, separators=(",", ":"))
    return (
        "# AXIOM PowerShell completion. Add this script to your PowerShell profile.\n"
        "$script:AXIOMCompletion = ConvertFrom-Json -InputObject @'\n"
        + schema
        + "\n'@\n"
        + r'''Register-ArgumentCompleter -Native -CommandName axiom -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $tokens = @($commandAst.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.ToString() })
    if ($tokens.Count -gt 0 -and $tokens[-1] -eq $wordToComplete) {
        if ($tokens.Count -eq 1) { $tokens = @() }
        else { $tokens = @($tokens[0..($tokens.Count - 2)]) }
    }

    $node = $script:AXIOMCompletion
    foreach ($token in $tokens) {
        if ($token.StartsWith('-')) { continue }
        if ($null -ne $node.commands -and $node.commands.PSObject.Properties.Name -contains $token) {
            $node = $node.commands.$token
        } else {
            break
        }
    }

    $candidates = @()
    if ($null -ne $node.commands) { $candidates += @($node.commands.PSObject.Properties.Name) }
    if ($null -ne $node.options) { $candidates += @($node.options) }
    $candidates |
        Where-Object { $_ -like "$wordToComplete*" } |
        Sort-Object -Unique |
        ForEach-Object {
            [System.Management.Automation.CompletionResult]::new(
                $_, $_, 'ParameterValue', $_
            )
        }
}'''
        + "\n"
    )


@app.command("completion")
def shell_completion_script(
    shell: str = typer.Argument(..., help="Shell name: bash, zsh, fish, or powershell."),
) -> None:
    """Print a supported shell completion script without installing it."""
    normalized = shell.strip().casefold()
    supported = {"bash", "zsh", "fish", "powershell"}
    if normalized not in supported:
        _cli_error("Unsupported shell. Choose bash, zsh, fish, or powershell.")
        return
    command = get_command(app)
    if normalized == "powershell":
        typer.echo(_powershell_completion_script(command), nl=False)
        return
    completion_type = get_completion_class(normalized)
    if completion_type is None:
        _cli_error(f"Shell completion for {normalized} is not supported by the installed Click version.")
        return
    # Click 8.3+ uses (cli, ctx_args, prog_name, complete_var); older
    # versions expose (cli, ctx, complete_var). Select by the installed class
    # signature rather than passing a Typer/Click option as a variable name.
    parameters = inspect.signature(completion_type).parameters
    if "ctx_args" in parameters and "prog_name" in parameters:
        completion = completion_type(command, {}, "axiom", "_AXIOM_COMPLETE")
    else:
        completion = completion_type(
            command,
            click.Context(command, info_name="axiom"),
            "_AXIOM_COMPLETE",
        )
    script = completion.source()
    typer.echo(script, nl=False)


@app.command()
def version():
    """Show AXIOM version."""
    console.print(f"[bold cyan]AXIOM[/bold cyan] v{__version__}")


@mcp_app.command("permissions")
def mcp_permissions() -> None:
    """Show active MCP profile and per-tool access decisions."""
    try:
        policy = ToolPermissionStore().list()
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    profiles = {profile["name"]: profile for profile in policy["profiles"]}
    active = policy["active_profile"]
    allowed = set(profiles[active]["allowed_tools"])
    table = Table(title=f"MCP permissions · active profile: {active}")
    table.add_column("Tool", style="cyan")
    table.add_column("Access")
    table.add_column("Purpose")
    for tool_name, description in TOOL_DESCRIPTIONS.items():
        table.add_row(tool_name, "ALLOW" if tool_name in allowed else "DENY", description)
    console.print(table)


@mcp_app.command("allow")
def mcp_allow(
    tool_name: str,
    profile: str = typer.Option("default", "--profile", help="Profile to update."),
) -> None:
    """Explicitly allow one MCP tool in a profile."""
    try:
        policy = ToolPermissionStore()
        current = next((item for item in policy.list()["profiles"] if item["name"] == profile), None)
        if current is None:
            _cli_error(f"Unknown MCP profile: {profile}")
        allowed = sorted(set(current["allowed_tools"]) | {tool_name})
        policy.save_profile(profile, allowed)
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    console.print(f"[green]Allowed[/green] {tool_name} in profile {profile}.")


@mcp_app.command("revoke")
def mcp_revoke(
    tool_name: str,
    profile: str = typer.Option("default", "--profile", help="Profile to update."),
) -> None:
    """Revoke one MCP tool from a profile."""
    try:
        policy = ToolPermissionStore()
        current = next((item for item in policy.list()["profiles"] if item["name"] == profile), None)
        if current is None:
            _cli_error(f"Unknown MCP profile: {profile}")
        allowed = [item for item in current["allowed_tools"] if item != tool_name]
        policy.save_profile(profile, allowed)
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    console.print(f"[yellow]Revoked[/yellow] {tool_name} from profile {profile}.")


@mcp_profile_app.command("create")
def mcp_profile_create(profile: str) -> None:
    """Create a profile with no allowed tools."""
    try:
        ToolPermissionStore().create_profile(profile)
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    console.print(f"Created default-deny MCP profile [cyan]{profile}[/cyan].")


@mcp_profile_app.command("use")
def mcp_profile_use(profile: str) -> None:
    """Select the profile enforced by the MCP server."""
    try:
        ToolPermissionStore().set_active(profile)
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    console.print(f"Active MCP profile is now [cyan]{profile}[/cyan].")


@experiment_app.command("list")
def experiment_list(
    limit: int = typer.Option(20, min=1, max=1000, help="Maximum records to display."),
    offset: int = typer.Option(0, min=0, max=1_000_000, help="Number of matching records to skip."),
    query: str | None = typer.Option(None, "--query", help="Search names, models, datasets, notes, and tags."),
    tag: str | None = typer.Option(None, "--tag", help="Filter by an exact tag."),
    json_output: bool = typer.Option(False, "--json", help="Print records as JSON."),
) -> None:
    """List saved training plans without claiming they were executed."""
    try:
        records = ExperimentStore().list_page(limit=limit, offset=offset, query=query, tag=tag)["experiments"]
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    if json_output:
        console.print_json(json.dumps(records, ensure_ascii=False))
        return
    if not records:
        console.print("No saved experiments. Create a plan in the GUI to start history.")
        return

    table = Table(title="Saved experiment plans")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Created")
    for record in records:
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        table.add_row(
            str(record.get("id", ""))[:12],
            str(record.get("name", "Unnamed")),
            str(plan.get("method", record.get("requested_method", "unknown"))),
            str(record.get("status", "planned")),
            str(record.get("created_at", "")),
        )
    console.print(table)


@experiment_app.command("show")
def experiment_show(experiment_id: str) -> None:
    """Show a saved experiment plan by its full ID."""
    try:
        record = ExperimentStore().get(experiment_id)
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    if record is None:
        _cli_error(f"Experiment not found: {experiment_id}")
    console.print_json(json.dumps(record, ensure_ascii=False))


@experiment_app.command("update")
def experiment_update(
    experiment_id: str,
    name: str | None = typer.Option(None, "--name"),
    notes: str | None = typer.Option(None, "--notes"),
    clear_notes: bool = typer.Option(False, "--clear-notes", help="Remove saved notes."),
    tags: list[str] | None = typer.Option(None, "--tag", help="Replace tags; repeat the option for multiple tags."),
) -> None:
    """Update experiment labels without changing the saved plan snapshot."""
    store = ExperimentStore()
    try:
        current = store.get(experiment_id)
        if current is None:
            _cli_error(f"Experiment not found: {experiment_id}")
        updated = store.update_metadata(
            experiment_id,
            name=name if name is not None else current.get("name", ""),
            notes=None if clear_notes else notes if notes is not None else current.get("notes"),
            tags=tags if tags is not None else current.get("tags", []),
        )
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    if updated is None:
        _cli_error(f"Experiment not found: {experiment_id}")
    console.print_json(json.dumps({"experiment": updated, "plan_immutable": True}, ensure_ascii=False))


@experiment_app.command("delete")
def experiment_delete(
    experiment_id: str,
    yes: bool = typer.Option(False, "--yes", help="Delete without an interactive confirmation."),
) -> None:
    """Delete one saved plan record; no run artifacts are removed."""
    if not yes and not typer.confirm(f"Delete saved plan {experiment_id}?", default=False):
        console.print("Cancelled.")
        return
    try:
        removed = ExperimentStore().delete(experiment_id)
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))
    if removed is None:
        _cli_error(f"Experiment not found: {experiment_id}")
    console.print_json(json.dumps({"deleted": removed, "run_artifacts_deleted": False}, ensure_ascii=False))


def _resolve_gui_port(cli_port: int | None) -> int:
    """Resolve GUI port from explicit CLI input or a validated environment default."""
    if cli_port is not None:
        port = cli_port
    else:
        raw = os.environ.get("AXIOM_GUI_PORT", "40507").strip()
        try:
            port = int(raw)
        except ValueError:
            raise ValueError("AXIOM_GUI_PORT must be an integer from 0 to 65535.") from None
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("GUI port must be an integer from 0 to 65535.")
    return port


@app.command("gui")
@app.command("ui")
def start_gui(
    port: int | None = typer.Option(
        None,
        "--port",
        min=0,
        max=65535,
        help="Local port to use; defaults to AXIOM_GUI_PORT or 40507. Use 0 for an ephemeral port. Keep the port stable to retain this browser origin's local account and sign-in state.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the local Control Center in your browser.",
    ),
) -> None:
    """Start the local API and open the AXIOM Control Center."""
    try:
        port = _resolve_gui_port(port)
    except ValueError as exc:
        _cli_error(str(exc))
    webui_dir = Path(__file__).resolve().parent.parent / "webui"
    if not (webui_dir / "index.html").is_file():
        _cli_error("The bundled Control Center is missing. Build the web UI before running axiom gui.")

    try:
        import uvicorn
    except ImportError as exc:
        _cli_error(f"The local API server dependency is unavailable: {exc}")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(128)
    except OSError as exc:
        listener.close()
        _cli_error(f"Could not start the local Control Center: {exc}")

    selected_port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{selected_port}"
    config = uvicorn.Config(
        "axiom.api.server:create_app",
        factory=True,
        host="127.0.0.1",
        port=selected_port,
        access_log=False,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, name="axiom-gui", daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    ready = False
    last_request_id = "not available"
    while thread.is_alive() and time.monotonic() < deadline:
        last_request_id = uuid.uuid4().hex
        health_request = urllib.request.Request(
            f"{base_url}/api/health",
            headers={"X-Request-ID": last_request_id},
        )
        try:
            with urllib.request.urlopen(health_request, timeout=0.5) as response:
                ready = response.status == 200
                if ready:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)

    if not ready:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        _cli_error(f"The local API did not become ready. Check the AXIOM installation and try again (request ID {last_request_id}).")

    ui_url = f"{base_url}/ui/"
    console.print(f"[bold green]AXIOM Control Center ready[/bold green]  {ui_url}")
    console.print("The local API is bound to loopback. Press Ctrl-C to stop it.")
    if open_browser:
        webbrowser.open(ui_url, new=2)

    try:
        thread.join()
    except KeyboardInterrupt:
        console.print("\nStopping the AXIOM Control Center…")
        server.should_exit = True
        thread.join(timeout=5)
    finally:
        listener.close()


@app.command()
def daemon(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Loopback address to bind. Use --allow-network for a non-local host.",
    ),
    port: int = typer.Option(
        0,
        "--port",
        min=0,
        max=65535,
        help="TCP port to bind; 0 chooses an available port automatically.",
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Allow binding beyond the local machine. Keep the default for private use.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON readiness and stopped records for a supervisor or script.",
    ),
) -> None:
    """Run AXIOM's local daemon and print how to inspect it."""

    try:
        run_daemon(
            host=host,
            port=port,
            allow_network=allow_network,
            machine_readable=json_output,
        )
    except (OSError, ValueError) as exc:
        _cli_error(str(exc))


@app.command()
def init(name: str):
    """Create a new AXIOM AI project."""
    try:
        root = create_project(name, Path.cwd())
    except (FileExistsError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold green]✓ AXIOM project created[/bold green]\n\n"
            f"[cyan]{root}[/cyan]\n\n"
            "Next steps:\n"
            f"  cd {name}\n"
            "  axiom model list",
            title="AXIOM",
        )
    )


@hf_app.command("login")
def hf_login():
    """Authenticate AXIOM with Hugging Face."""
    console.print(
        Panel.fit(
            "[bold cyan]AXIOM HUGGING FACE LOGIN[/bold cyan]\n\n"
            "A Hugging Face login will be started.\n"
            "Your token is handled by Hugging Face's authentication system\n"
            "and is not written into the AXIOM repository.",
            title="AXIOM",
        )
    )

    try:
        login()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Hugging Face login failed: {exc}")
        raise typer.Exit(code=1)

    console.print("[green]✓[/green] Hugging Face authentication configured.")


@hf_app.command("status")
def hf_status():
    """Show Hugging Face authentication status."""
    try:
        user = whoami()
    except Exception as exc:  # noqa: BLE001
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if exc.__class__.__name__ == "LocalTokenNotFoundError" or status_code == 401:
            console.print(
                Panel.fit(
                    "[yellow]Not authenticated[/yellow]\n\nRun:\n  axiom hf login",
                    title="AXIOM",
                )
            )
            return

        _cli_error(f"Hugging Face status check failed: {exc}")

    name = user.get("name") or user.get("fullname") or user.get("username") or "Unknown"

    orgs = user.get("orgs") or []

    if isinstance(orgs, list):
        organization_names = [
            item.get("name", str(item)) if isinstance(item, dict) else str(item)
            for item in orgs
        ]
    else:
        organization_names = []

    console.print(
        Panel.fit(
            f"[bold green]✓ Authenticated[/bold green]\n\n"
            f"Account: {name}\n"
            f"Organizations: "
            f"{', '.join(organization_names) if organization_names else 'None'}\n\n"
            "[dim]Token value is never displayed.[/dim]",
            title="AXIOM",
        )
    )


@hf_app.command("logout")
def hf_logout():
    """Log out of Hugging Face."""
    try:
        logout()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Hugging Face logout failed: {exc}")
        raise typer.Exit(code=1)

    console.print("[green]✓[/green] Hugging Face authentication removed.")


@supercompress_app.command("status")
def supercompress_status():
    """Show SuperCompress configuration without exposing secrets."""
    configured = bool(os.getenv("SUPERCOMPRESS_API_KEY", "").strip())

    endpoint = redacted_base_url()

    key_status = (
        "[green]configured[/green]" if configured else "[red]not configured[/red]"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]AXIOM SUPERCOMPRESS[/bold cyan]\n\n"
            f"Endpoint: {endpoint}\n"
            f"API key:  {key_status}\n\n"
            "[dim]The secret value is never displayed.[/dim]",
            title="AXIOM",
        )
    )


@supercompress_app.command("compress")
def supercompress_compress(
    context_file: str,
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
    ),
    budget_ratio: float = typer.Option(
        0.35,
        "--budget-ratio",
        min=0.05,
        max=1.0,
    ),
    ccr: bool = typer.Option(
        False,
        "--ccr",
        help="Enable reversible Cache-Compress-Retrieve mode.",
    ),
):
    """Compress a context file before inference."""
    path = Path(context_file)

    if not path.is_file():
        console.print(f"[red]Error:[/red] Context file not found: {path}")
        raise typer.Exit(code=1)

    try:
        context = path.read_text(encoding="utf-8")

        result = compress_context(
            context,
            query,
            budget_ratio=budget_ratio,
            ccr=ccr,
        )

    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        _cli_error(str(exc))

    savings = (
        f"{result.savings_pct:.1f}%" if result.savings_pct is not None else "Unknown"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]AXIOM + SUPERCOMPRESS[/bold cyan]\n\n"
            f"Original tokens: "
            f"{result.original_tokens or 'Unknown'}\n"
            f"Kept tokens:     "
            f"{result.kept_tokens or 'Unknown'}\n"
            f"Tokens saved:    "
            f"{result.tokens_saved or 'Unknown'}\n"
            f"Reduction:       {savings}\n"
            f"Risk:             "
            f"{result.compression_risk or 'Unknown'}\n"
            f"Policy:           "
            f"{result.policy_name or 'Unknown'}\n"
            f"Mode:             "
            f"{result.mode or 'Unknown'}\n\n"
            f"{result.compressed_text}",
            title="AXIOM",
        )
    )


@integration_app.command("list")
def integration_list(
    json_output: bool = typer.Option(False, "--json", help="Emit integration records as JSON."),
) -> None:
    """List configured AI agent integrations."""
    registry = IntegrationRegistry()
    integrations = registry.list()
    if json_output:
        typer.echo(json.dumps({
            "schema_version": 1,
            "total": len(integrations),
            "integrations": [item.to_dict() for item in integrations],
        }, ensure_ascii=False, sort_keys=True))
        return

    if not integrations:
        console.print("[yellow]No agent integrations configured.[/yellow]")
        return

    table = Table(title="AXIOM Agent Integrations")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Target")
    table.add_column("Description")

    for item in integrations:
        target = item.command or item.url or "-"
        table.add_row(item.name, item.type.value, target, item.description or "-")
    console.print(table)

@integration_app.command("add")
def integration_add(
    name: str,
    type: str = typer.Option(
        "mcp",
        "--type",
        help="Integration type: mcp, cli, http, or a2a.",
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        help="Command used by CLI/MCP integrations.",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        help="URL used by HTTP/A2A integrations.",
    ),
):
    """Register an external AI agent or runtime."""
    try:
        integration_type = IntegrationType(type.lower())
    except ValueError:
        console.print("[red]Error:[/red] Type must be one of: mcp, cli, http, a2a.")
        raise typer.Exit(code=1)

    if (
        integration_type
        in {
            IntegrationType.MCP,
            IntegrationType.CLI,
        }
        and not command
    ):
        console.print(
            "[red]Error:[/red] --command is required for "
            f"{integration_type.value} integrations."
        )
        raise typer.Exit(code=1)

    if (
        integration_type
        in {
            IntegrationType.HTTP,
            IntegrationType.A2A,
        }
        and not url
    ):
        console.print(
            "[red]Error:[/red] --url is required for "
            f"{integration_type.value} integrations."
        )
        raise typer.Exit(code=1)

    registry = IntegrationRegistry()

    try:
        registry.add(
            AgentIntegration(
                name=name,
                type=integration_type,
                command=command,
                url=url,
                description=f"AXIOM {integration_type.value} integration",
            )
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Registered integration: [cyan]{name}[/cyan]")


@integration_app.command("remove")
def integration_remove(name: str):
    """Remove an agent integration."""
    registry = IntegrationRegistry()

    if not registry.remove(name):
        console.print(f"[yellow]Integration not found:[/yellow] {name}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Removed integration: [cyan]{name}[/cyan]")


@integration_app.command("doctor")
def integration_doctor():
    """Check whether configured integration commands exist."""
    import shutil

    registry = IntegrationRegistry()
    integrations = registry.list()

    if not integrations:
        console.print("[yellow]No agent integrations configured.[/yellow]")
        return

    table = Table(title="AXIOM Integration Doctor")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Target")

    for item in integrations:
        if item.command:
            executable = item.command.split()[0]
            available = shutil.which(executable) is not None

            status = (
                "[green]✓ Available[/green]" if available else "[red]✗ Not found[/red]"
            )
        elif item.url:
            status = "[cyan]Configured[/cyan]"
        else:
            status = "[yellow]Incomplete[/yellow]"

        table.add_row(
            item.name,
            item.type.value,
            status,
            item.command or item.url or "-",
        )

    console.print(table)


@system_app.command("info")
def system_info(
    json_output: bool = typer.Option(False, "--json", help="Emit detected hardware and fit estimates as JSON."),
):
    """Show detected hardware and AI runtime capability."""
    hardware = detect_hardware()
    fit_estimates = []
    if hardware.vram_gb is not None:
        for billions in (1, 3, 7, 8, 14, 32, 70):
            fit_estimates.append({
                "parameters_billions": billions,
                "precision_candidates": "FP16 / Q4",
                "assessment": estimate_model_fit(billions, hardware),
            })
    if json_output:
        typer.echo(json.dumps({
            "schema_version": 1,
            "hardware": {
                "os_name": hardware.os_name,
                "architecture": hardware.architecture,
                "cpu_cores": hardware.cpu_cores,
                "ram_gb": hardware.ram_gb,
                "gpu_name": hardware.gpu_name,
                "vram_gb": hardware.vram_gb,
                "cuda_available": hardware.cuda_available,
            },
            "model_fit_estimates": fit_estimates,
            "fit_scope_note": "Fit values are heuristic screening estimates, not measured runtime capacity or benchmark results.",
        }, ensure_ascii=False, sort_keys=True))
        return

    gpu = hardware.gpu_name or "Not detected"
    vram = f"{hardware.vram_gb:.2f} GB" if hardware.vram_gb is not None else "N/A"
    console.print(Panel.fit(
        f"[bold cyan]AXIOM HARDWARE[/bold cyan]\n\n"
        f"OS:              {hardware.os_name}\n"
        f"Architecture:    {hardware.architecture}\n"
        f"CPU cores:       {hardware.cpu_cores}\n"
        f"RAM:             {hardware.ram_gb:.2f} GB\n"
        f"GPU:             {gpu}\n"
        f"VRAM:            {vram}\n"
        f"CUDA available:  {'✓' if hardware.cuda_available else '✗'}",
        title="AXIOM",
    ))
    if fit_estimates:
        table = Table(title="Model Fit Estimates")
        table.add_column("Model")
        table.add_column("FP16 / Q4")
        for item in fit_estimates:
            table.add_row(f"{item['parameters_billions']}B", item["assessment"])
        console.print(table)

@train_app.command("plan")
def train_plan(
    parameters: float = typer.Argument(
        ..., help="Model size in billions of parameters."
    ),
    method: str = typer.Option(
        "auto",
        "--method",
        help="Training method: auto, qlora, lora, or full.",
    ),
    save: bool = typer.Option(False, "--save", help="Save this exact planning snapshot to local experiment history."),
    name: str | None = typer.Option(None, "--name", help="Saved plan name; requires --save."),
    model: str | None = typer.Option(None, "--model", help="Optional model label to save with the plan."),
    dataset: str | None = typer.Option(None, "--dataset", help="Optional dataset path or label to save with the plan."),
    notes: str | None = typer.Option(None, "--notes", help="Optional note to save with the plan."),
    tags: list[str] | None = typer.Option(None, "--tag", help="Saved tag; repeat for multiple tags. Requires --save."),
    json_output: bool = typer.Option(False, "--json", help="Emit the planning result and optional saved snapshot as JSON."),
):
    """Generate and optionally save a hardware-aware plan without starting training."""
    if not save and any(value is not None for value in (name, model, dataset, notes, tags)):
        _cli_error("--name, --model, --dataset, --notes, and --tag require --save.")

    hardware = detect_hardware()

    try:
        plan = create_training_plan(
            parameter_billions=parameters,
            hardware=hardware,
            method=method.lower(),
        )
    except ValueError as exc:
        _cli_error(str(exc))

    saved_record = None
    if save:
        saved_record = {
            "schema_version": 1,
            "id": uuid.uuid4().hex,
            "name": name or f"{parameters:g}B {plan.method} plan",
            "created_at": datetime.now(UTC).isoformat(),
            "model": model,
            "dataset": dataset,
            "notes": notes,
            "tags": tags or [],
            "parameter_billions": parameters,
            "requested_method": method.lower(),
            "status": "planned",
            "hardware": hardware.__dict__,
            "plan": plan.__dict__,
        }
        try:
            ExperimentStore().add(saved_record)
        except (OSError, TimeoutError, ValueError) as exc:
            _cli_error(f"Plan could not be saved: {exc}")

    if json_output:
        console.print_json(json.dumps({
            "status": "planning_only",
            "executed": False,
            "hardware": hardware.__dict__,
            "plan": plan.__dict__,
            "experiment": saved_record,
        }, ensure_ascii=False, allow_nan=False))
        return

    if hardware.gpu_name:
        gpu = hardware.gpu_name
        vram = (
            f"{hardware.vram_gb:.2f} GB" if hardware.vram_gb is not None else "Unknown"
        )
    else:
        gpu = "Not detected"
        vram = "N/A"

    if plan.fits_hardware is None:
        fit = "[yellow]? UNKNOWN[/yellow]"
    elif plan.fits_hardware:
        fit = "[green]✓ FITS[/green]"
    else:
        fit = "[red]✗ DOES NOT FIT[/red]"
    estimated_vram = (
        f"{plan.estimated_vram_gb:.2f} GB"
        if plan.estimated_vram_gb is not None
        else "Unknown"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]AXIOM TRAINING PLANNER[/bold cyan]\n\n"
            f"Model:              {parameters:g}B parameters\n"
            f"Method:             {plan.method}\n\n"
            f"Hardware\n"
            f"GPU:                {gpu}\n"
            f"VRAM:               {vram}\n\n"
            f"Recommended\n"
            f"Precision:           {plan.precision}\n"
            f"LoRA rank:          {plan.lora_rank}\n"
            f"Batch size:         {plan.batch_size}\n"
            f"Grad accumulation:  {plan.gradient_accumulation}\n"
            f"Sequence length:    {plan.sequence_length}\n"
            f"Learning rate:      {plan.learning_rate:g}\n\n"
            f"Estimated VRAM:     {estimated_vram}\n"
            f"Hardware check:     {fit}\n\n"
            f"{plan.reason}",
            title="AXIOM",
        )
    )
    if saved_record:
        console.print(f"Saved plan snapshot [cyan]{saved_record['id']}[/cyan]. No training job was started.")


@model_app.command("list")
def model_list(
    json_output: bool = typer.Option(False, "--json", help="Emit registered model records as JSON."),
) -> None:
    """List registered models."""
    registry = ModelRegistry()
    models = registry.list()
    if json_output:
        typer.echo(json.dumps({
            "schema_version": 1,
            "total": len(models),
            "models": [model.to_dict() for model in models],
        }, ensure_ascii=False, sort_keys=True))
        return

    if not models:
        console.print("[yellow]No models registered.[/yellow]")
        return

    table = Table(title="AXIOM Models")
    table.add_column("Name")
    table.add_column("Source")
    table.add_column("Format")
    table.add_column("Parameters")
    table.add_column("Quantization")
    for model in models:
        table.add_row(model.name, model.source, model.format, model.parameters or "-", model.quantization or "-")
    console.print(table)

@model_app.command("export")
def model_export(
    path: str = typer.Option("models.registry.json", "--path", help="Destination JSON file."),
    json_output: bool = typer.Option(False, "--json", help="Emit the export operation result as JSON."),
) -> None:
    """Export registered model metadata to a versioned JSON file."""
    registry = ModelRegistry()
    target = Path(path).expanduser()
    try:
        if target.resolve(strict=False) == registry.registry_file.resolve(strict=False):
            raise ValueError("Export destination must not be the live model registry file.")
        models = registry.list()
        payload = {
            "schema_version": 1,
            "models": [model.to_dict() for model in models],
            "total": len(models),
            "exported_at": datetime.now(UTC).isoformat(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_export_failed", "message": str(exc)}}, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps({"schema_version": 1, "exported": True, "path": str(target.resolve()), "total": len(models)}, sort_keys=True))
        return
    console.print(f"[green]✓[/green] Exported {len(models)} model record(s) to {target}")


@model_app.command("import")
def model_import(
    path: str,
    json_output: bool = typer.Option(False, "--json", help="Emit imported model records as JSON."),
) -> None:
    """Import up to 1000 registered model metadata records from versioned JSON."""
    source = Path(path).expanduser()
    try:
        if not source.is_file():
            raise ValueError("Model import path must be an existing file.")
        if source.stat().st_size > ModelRegistry.MAX_BYTES:
            raise ValueError("Model import exceeds the 4 MB size limit.")
        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            values = raw
        elif isinstance(raw, dict) and raw.get("schema_version") == 1 and isinstance(raw.get("models"), list):
            values = raw["models"]
        else:
            raise ValueError("Model import must be a JSON array or a schema_version 1 export object.")
        registry = ModelRegistry()
        imported = registry.import_models(values)
        after_count = len(registry.list())
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_import_failed", "message": str(exc)}}, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] Model import failed: {exc}")
        raise typer.Exit(code=1) from exc
    payload = {
        "schema_version": 1,
        "imported": len(imported),
        "total_models_after_import": after_count,
        "models": [model.to_dict() for model in imported],
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        console.print(f"[green]✓[/green] Imported {len(imported)} model record(s). Registry now contains {after_count} model(s).")


@model_app.command("remove")
def model_remove(
    model_id: str,
    json_output: bool = typer.Option(False, "--json", help="Emit the removed record as JSON."),
) -> None:
    """Remove one model record by its stable registry ID."""
    registry = ModelRegistry()
    try:
        removed = registry.remove(model_id)
    except (OSError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_remove_failed", "message": str(exc)}}, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] Could not remove model record: {exc}")
        raise typer.Exit(code=1) from exc
    if removed is None:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_not_found", "message": "No registered model has that ID."}}, sort_keys=True))
        else:
            console.print("[yellow]No registered model has that ID.[/yellow]")
        raise typer.Exit(code=1)
    payload = {"schema_version": 1, "removed": True, "model": removed.to_dict()}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        console.print(f"[green]✓[/green] Removed model: {removed.name} ({removed.id})")


@model_app.command("favorite")
def model_favorite(
    model_id: str,
    favorite: bool = typer.Option(True, "--favorite/--unfavorite", help="Set or clear the model's favorite flag."),
    json_output: bool = typer.Option(False, "--json", help="Emit the updated model record as JSON."),
) -> None:
    """Mark a registered model as favorite or clear its favorite flag."""
    registry = ModelRegistry()
    try:
        updated = registry.update(model_id, {"favorite": favorite})
    except (OSError, TypeError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_favorite_failed", "message": str(exc)}}, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] Could not update model favorite state: {exc}")
        raise typer.Exit(code=1) from exc
    if updated is None:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_not_found", "message": "No registered model has that ID."}}, sort_keys=True))
        else:
            console.print("[yellow]No registered model has that ID.[/yellow]")
        raise typer.Exit(code=1)
    payload = {"schema_version": 1, "updated": True, "model": updated.to_dict()}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        action = "Favorited" if favorite else "Unfavorited"
        console.print(f"[green]✓[/green] {action} model: {updated.name}")


@model_app.command("info")
def model_info(
    repo_id: str,
    json_output: bool = typer.Option(False, "--json", help="Emit bounded Hugging Face metadata as JSON."),
) -> None:
    """Show metadata for a Hugging Face model without downloading weights."""
    api = HfApi()
    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001
        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "error": {"code": "model_metadata_unavailable", "message": "Could not retrieve model information."},
            }, sort_keys=True))
        else:
            console.print("[red]Error:[/red] Could not retrieve model information.")
        raise typer.Exit(code=1)

    files = getattr(info, "siblings", None) or []
    file_names = [name for item in files if (name := getattr(item, "rfilename", None))]
    formats: set[str] = set()
    for name in file_names:
        lower = name.lower()
        if lower.endswith(".safetensors"):
            formats.add("Safetensors")
        elif lower.endswith(".gguf"):
            formats.add("GGUF")
        elif lower.endswith((".bin", ".pt", ".pth")):
            formats.add("PyTorch")

    tags = getattr(info, "tags", None) or []
    library = getattr(info, "library_name", None) or "Unknown"
    pipeline = getattr(info, "pipeline_tag", None) or "Unknown"
    downloads = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0
    gated = getattr(info, "gated", False)
    private = getattr(info, "private", False)
    author = getattr(info, "author", None) or "Unknown"

    if json_output:
        payload = {
            "schema_version": 1,
            "model": {
                "id": info.id,
                "author": author,
                "library": library,
                "task": pipeline,
                "formats": sorted(formats),
                "downloads": downloads,
                "likes": likes,
                "gated": gated,
                "private": private,
                "files": {"items": file_names[:500], "total": len(file_names), "truncated": len(file_names) > 500},
                "tags": {"items": tags[:100], "total": len(tags), "truncated": len(tags) > 100},
            },
            "scope_note": "Metadata only; no model weights were downloaded.",
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    format_text = ", ".join(sorted(formats)) if formats else "Not detected"
    console.print(Panel.fit(
        f"[bold cyan]AXIOM HUGGING FACE MODEL[/bold cyan]\n\n"
        f"Repository:   {info.id}\n"
        f"Author:       {author}\n"
        f"Library:      {library}\n"
        f"Task:         {pipeline}\n"
        f"Formats:      {format_text}\n"
        f"Downloads:    {downloads:,}\n"
        f"Likes:        {likes:,}\n"
        f"Gated:        {'✓' if gated else '✗'}\n"
        f"Private:      {'✓' if private else '✗'}\n\n"
        f"Files:        {len(file_names):,}\n"
        f"Tags:         {', '.join(tags[:8]) if tags else '-'}\n\n"
        "Metadata only — no model weights were downloaded.",
        title="AXIOM",
    ))

@model_app.command("add-hf")
def model_add_hf(repo_id: str):
    """Discover and register a Hugging Face model."""
    api = HfApi()

    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Could not retrieve model information: {exc}")
        raise typer.Exit(code=1)

    files = getattr(info, "siblings", None) or []

    file_names = [getattr(item, "rfilename", "") for item in files]

    formats = set()

    for name in file_names:
        lower = name.lower()

        if lower.endswith(".safetensors"):
            formats.add("safetensors")
        elif lower.endswith(".gguf"):
            formats.add("gguf")
        elif lower.endswith((".bin", ".pt", ".pth")):
            formats.add("pytorch")

    model_format = ", ".join(sorted(formats)) if formats else "unknown"

    pipeline = getattr(info, "pipeline_tag", None)

    registry = ModelRegistry()

    try:
        registry.add(
            Model(
                name=repo_id,
                source=f"huggingface:{repo_id}",
                format=model_format,
                parameters=None,
                quantization=None,
            )
        )
    except ValueError:
        console.print(f"[yellow]Model already registered:[/yellow] {repo_id}")
        return

    console.print(
        Panel.fit(
            f"[bold green]✓ Model discovered and registered[/bold green]\n\n"
            f"Model:    [cyan]{repo_id}[/cyan]\n"
            f"Task:     {pipeline or 'Unknown'}\n"
            f"Format:   {model_format}\n\n"
            "[dim]Weights were not downloaded.[/dim]",
            title="AXIOM",
        )
    )


@model_app.command("add")
def model_add(
    name: str,
    source: str,
    format: str = "unknown",
    parameters: str | None = None,
    quantization: str | None = None,
    architecture: str | None = typer.Option(None, "--architecture", help="Known model architecture."),
    license_name: str | None = typer.Option(None, "--license", help="Source/license identifier from trusted metadata."),
    tag: list[str] | None = typer.Option(None, "--tag", help="Repeat for each model tag."),
    content_sha256: str | None = typer.Option(None, "--content-sha256", help="Optional 64-hex digest for inspected local weights."),
    json_output: bool = typer.Option(False, "--json", help="Emit the created model record as JSON."),
):
    """Register model metadata without downloading or moving weights."""
    registry = ModelRegistry()
    try:
        model = Model(
            name=name,
            source=source,
            format=format,
            parameters=parameters,
            quantization=quantization,
            architecture=architecture,
            license=license_name,
            tags=tag or [],
            content_sha256=content_sha256,
        )
        registry.add(model)
    except (TypeError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_add_failed", "message": str(exc)}}, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps({"schema_version": 1, "created": True, "model": model.to_dict()}, ensure_ascii=False, sort_keys=True))
        return
    console.print(f"[green]✓[/green] Registered model: [cyan]{name}[/cyan]")


@model_app.command("update")
def model_update(
    model_id: str,
    name: str | None = typer.Option(None, "--name"),
    source: str | None = typer.Option(None, "--source"),
    format: str | None = typer.Option(None, "--format"),
    parameters: str | None = typer.Option(None, "--parameters"),
    quantization: str | None = typer.Option(None, "--quantization"),
    architecture: str | None = typer.Option(None, "--architecture"),
    license_name: str | None = typer.Option(None, "--license"),
    tag: list[str] | None = typer.Option(None, "--tag", help="Replace all tags; repeat for each tag."),
    clear_tags: bool = typer.Option(False, "--clear-tags", help="Clear all tags."),
    content_sha256: str | None = typer.Option(None, "--content-sha256"),
    favorite: bool | None = typer.Option(None, "--favorite/--unfavorite"),
    json_output: bool = typer.Option(False, "--json", help="Emit the updated model record as JSON."),
):
    """Update validated registry metadata without touching model weights."""
    changes: dict[str, object] = {}
    for key, value in (
        ("name", name),
        ("source", source),
        ("format", format),
        ("parameters", parameters),
        ("quantization", quantization),
        ("architecture", architecture),
        ("license", license_name),
        ("content_sha256", content_sha256),
    ):
        if value is not None:
            changes[key] = value
    if tag is not None:
        changes["tags"] = tag
    if clear_tags:
        if tag is not None:
            message = "--clear-tags cannot be combined with --tag."
            if json_output:
                typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_update_invalid", "message": message}}, sort_keys=True))
            else:
                console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(code=1)
        changes["tags"] = []
    if favorite is not None:
        changes["favorite"] = favorite
    if not changes:
        message = "Specify at least one metadata option to update."
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_update_empty", "message": message}}, sort_keys=True))
        else:
            console.print(f"[yellow]No model changes requested.[/yellow] {message}")
        raise typer.Exit(code=1)

    registry = ModelRegistry()
    try:
        updated = registry.update(model_id, changes)
    except (OSError, TypeError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_update_failed", "message": str(exc)}}, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] Could not update model metadata: {exc}")
        raise typer.Exit(code=1) from exc
    if updated is None:
        message = "No registered model has that ID."
        if json_output:
            typer.echo(json.dumps({"schema_version": 1, "error": {"code": "model_not_found", "message": message}}, sort_keys=True))
        else:
            console.print(f"[yellow]{message}[/yellow]")
        raise typer.Exit(code=1)
    payload = {"schema_version": 1, "updated": True, "model": updated.to_dict()}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        console.print(f"[green]✓[/green] Updated model metadata: {updated.name}")

@model_app.command("analyze")
def model_analyze(
    repo_id: str,
    json_output: bool = typer.Option(False, "--json", help="Emit versioned model analysis as JSON."),
):
    """Analyze Hugging Face model metadata/config without downloading model weights."""
    api = HfApi()
    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001
        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "error": {"code": "model_analysis_unavailable", "message": "Could not retrieve model metadata."},
            }, sort_keys=True))
        else:
            console.print("[red]Error:[/red] Could not retrieve model information.")
        raise typer.Exit(code=1)

    siblings = getattr(info, "siblings", None) or []
    files = [
        name
        for item in siblings
        if (name := getattr(item, "rfilename", None))
    ]
    config = {}
    warnings = []
    try:
        config_file = api.hf_hub_download(repo_id=repo_id, filename="config.json")
        config_path = Path(config_file)
        if config_path.is_file():
            loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded_config, dict):
                config = loaded_config
            else:
                warnings.append("config.json did not contain a JSON object; architecture-derived values are unavailable.")
        else:
            warnings.append("config.json was not found in the metadata cache; analysis may be incomplete.")
    except Exception:  # noqa: BLE001
        warnings.append("Could not read config.json; analysis may be incomplete.")

    total_size = 0
    sizes_known = 0
    for item in siblings:
        name = getattr(item, "rfilename", "") or ""
        size = getattr(item, "size", None)
        if size is not None and name.lower().endswith((".safetensors", ".gguf", ".bin", ".pt", ".pth")):
            try:
                total_size += int(size)
                sizes_known += 1
            except (TypeError, ValueError):
                warnings.append("Some checkpoint size metadata was not numeric.")
    analysis = analyze_config(
        repo_id=repo_id,
        config=config,
        files=files,
        weight_size_bytes=total_size if sizes_known else None,
    )

    if json_output:
        payload = {
            "schema_version": 1,
            "analysis": {
                "repo_id": analysis.repo_id,
                "architecture": analysis.architecture,
                "model_type": analysis.model_type,
                "parameter_count": analysis.parameter_count,
                "parameter_source": analysis.parameter_source,
                "context_length": analysis.context_length,
                "precision": analysis.precision,
                "weight_size_gb": analysis.weight_size_gb,
                "file_count": analysis.file_count,
                "files": {
                    "items": analysis.files[:500],
                    "total": analysis.file_count,
                    "truncated": analysis.file_count > 500,
                },
            },
            "warnings": warnings,
            "scope_note": "Metadata/config analysis only; no model weights were downloaded. Parameter counts and precision can be inferred from configuration metadata and are not runtime measurements.",
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    if warnings:
        for warning in warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    parameters = f"{analysis.parameter_count / 1_000_000_000:.2f}B" if analysis.parameter_count else "Unknown"
    context = f"{analysis.context_length:,}" if analysis.context_length else "Unknown"
    weight_size = f"{analysis.weight_size_gb:.2f} GB" if analysis.weight_size_gb is not None else "Unknown"
    console.print(Panel.fit(
        f"[bold cyan]AXIOM MODEL ANALYSIS[/bold cyan]\n\n"
        f"Repository:      {analysis.repo_id}\n"
        f"Architecture:    {analysis.architecture}\n"
        f"Model type:      {analysis.model_type}\n"
        f"Parameters:      {parameters}\n"
        f"Parameter basis: {analysis.parameter_source}\n"
        f"Context length:  {context}\n"
        f"Precision:       {analysis.precision}\n"
        f"Weight size:     {weight_size}\n"
        f"Files:           {analysis.file_count:,}\n\n"
        "Metadata/config analysis only — no model weights were downloaded.",
        title="AXIOM",
    ))

@model_app.command("pull")
def model_pull(
    repo_id: str,
    max_disk_usage_gb: float = typer.Option(
        85.0,
        "--max-disk-usage",
        help="Maximum allowed disk usage percentage.",
    ),
):
    """Safely download a Hugging Face model after disk checks."""
    api = HfApi()

    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Could not retrieve model information: {exc}")
        raise typer.Exit(code=1)

    total_size = 0

    for item in getattr(info, "siblings", None) or []:
        size = getattr(item, "size", None)

        if size is not None:
            total_size += int(size)

    free_bytes, total_bytes = disk_info(".")

    used_bytes = total_bytes - free_bytes
    used_percent = (used_bytes / total_bytes) * 100 if total_bytes else 100
    download_gb = total_size / (1024**3)
    free_gb = free_bytes / (1024**3)
    remaining_gb = free_gb - download_gb

    console.print(
        Panel.fit(
            f"[bold cyan]AXIOM MODEL DOWNLOAD CHECK[/bold cyan]\n\n"
            f"Model:             {repo_id}\n"
            f"Repository size:   {download_gb:.2f} GB\n"
            f"Free disk:         {free_gb:.2f} GB\n"
            f"Free after pull:   {remaining_gb:.2f} GB\n"
            f"Current usage:     {used_percent:.1f}%\n"
            f"Usage limit:       {max_disk_usage_gb:.1f}%",
            title="AXIOM",
        )
    )

    if remaining_gb < 0:
        console.print("[red]✗ Download blocked:[/red] insufficient disk space.")
        raise typer.Exit(code=1)

    projected_used_percent = (
        ((total_bytes - free_bytes + total_size) / total_bytes) * 100
        if total_bytes
        else 100
    )

    if projected_used_percent > max_disk_usage_gb:
        console.print(
            "[red]✗ Download blocked:[/red] "
            f"projected disk usage would reach {projected_used_percent:.1f}%."
        )
        raise typer.Exit(code=1)

    console.print("[green]✓ Disk safety check passed.[/green]")

    try:
        destination = snapshot_download(
            repo_id=repo_id,
            local_dir=Path("models") / repo_id.replace("/", "__"),
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Model download failed: {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓ Model downloaded:[/green] {destination}")


@model_app.command("inspect")
def model_inspect(
    path: str,
    json_output: bool = typer.Option(False, "--json", help="Emit the complete local model inspection as JSON."),
) -> None:
    """Inspect a local AI model directory."""
    try:
        result = inspect_model(path)
    except (FileNotFoundError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "error": {"code": "model_inspection_failed", "message": str(exc)},
            }, ensure_ascii=False, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps({
            "schema_version": 1,
            "inspection": {
                "path": str(result.path),
                "architecture": result.architecture,
                "model_type": result.model_type,
                "parameter_count": result.parameter_count,
                "parameter_source": result.parameter_source,
                "model_format": result.model_format,
                "weight_size_bytes": result.weight_size_bytes,
                "has_config": result.has_config,
                "has_tokenizer": result.has_tokenizer,
                "has_safetensors": result.has_safetensors,
                "has_gguf": result.has_gguf,
                "has_pytorch_weights": result.has_pytorch_weights,
                "capabilities": result.capabilities,
                "estimated_vram_gb": result.estimated_vram_gb,
                "status": result.status,
            },
        }, ensure_ascii=False, sort_keys=True))
        return

    parameters = f"{result.parameter_count / 1_000_000_000:.2f}B" if result.parameter_count else "Unknown"
    size_gb = result.weight_size_bytes / (1024**3)
    vram = f"{result.estimated_vram_gb:.2f} GB" if result.estimated_vram_gb is not None else "Unknown"
    console.print(Panel.fit(
        f"[bold cyan]AXIOM MODEL INSPECTION[/bold cyan]\n\n"
        f"Path:           {result.path}\n"
        f"Architecture:   {result.architecture or 'Unknown'}\n"
        f"Model type:     {result.model_type or 'Unknown'}\n"
        f"Format:         {result.model_format}\n"
        f"Parameters:     {parameters}\n"
        f"Parameter basis: {result.parameter_source}\n"
        f"Weight size:    {size_gb:.2f} GB\n"
        f"Estimated VRAM: {vram}\n\n"
        f"Config:         {'✓' if result.has_config else '✗'}\n"
        f"Tokenizer:      {'✓' if result.has_tokenizer else '✗'}\n"
        f"Safetensors:    {'✓' if result.has_safetensors else '✗'}\n"
        f"GGUF:           {'✓' if result.has_gguf else '✗'}\n"
        f"PyTorch:        {'✓' if result.has_pytorch_weights else '✗'}\n\n"
        f"Capabilities:   {', '.join(result.capabilities)}\n\n"
        f"Status:         {result.status}",
        title="AXIOM",
    ))

@dataset_app.command("inspect")
def dataset_inspect(
    path: str,
    json_output: bool = typer.Option(False, "--json", help="Emit the complete dataset inspection as JSON."),
) -> None:
    """Inspect a dataset and report its quality."""
    try:
        result = inspect_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "error": {"code": "dataset_inspection_failed", "message": str(exc)},
            }, ensure_ascii=False, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(dataset_inspection_payload(result), ensure_ascii=False, sort_keys=True))
        return

    console.print(Panel.fit(
        f"[bold cyan]AXIOM DATASET INSPECTION[/bold cyan]\n\n"
        f"File: {result.path}\n"
        f"Format: {result.format}\n\n"
        f"Samples:          {result.samples:,}\n"
        f"Valid:            {result.valid:,}\n"
        f"Invalid:          {result.invalid:,}\n"
        f"Empty:            {result.empty:,}\n"
        f"Duplicates:       {result.duplicates:,}\n"
        f"Estimated tokens: {result.estimated_tokens:,}\n\n"
        f"Fields: {', '.join(result.fields) if result.fields else '-'}\n"
        f"Invalid records: {result.invalid:,}",
        title="AXIOM",
    ))
    for line_number, message in result.errors:
        console.print(f"[red]INVALID[/red] line {line_number}: {message}")

@dataset_app.command("clean")
def dataset_clean(
    path: str,
    output: str | None = None,
    json_output: bool = typer.Option(False, "--json", help="Emit cleaning counts as JSON."),
):
    """Clean a JSONL dataset without modifying the original."""
    source = Path(path)

    def report_error(code: str, message: str) -> None:
        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "error": {"code": code, "message": message},
            }, ensure_ascii=False, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] {message}")

    if not source.exists():
        report_error("dataset_not_found", "Dataset not found.")
        raise typer.Exit(code=1)
    if not source.is_file():
        report_error("dataset_not_file", "Dataset path is not a file.")
        raise typer.Exit(code=1)
    if source.suffix.lower() != ".jsonl":
        report_error("unsupported_dataset_format", "Cleaning currently supports .jsonl files.")
        raise typer.Exit(code=1)

    destination = Path(output) if output else source.with_name(f"{source.stem}.cleaned.jsonl")
    try:
        result = clean_jsonl(source, destination)
    except OSError as exc:
        report_error("dataset_cleaning_failed", str(exc))
        raise typer.Exit(code=1) from exc

    payload = {
        "schema_version": 1,
        "cleaning": {
            "source": str(result.source),
            "output": str(result.output),
            "total_lines": result.total_lines,
            "kept": result.kept,
            "removed_invalid": result.removed_invalid,
            "removed_empty": result.removed_empty,
            "removed_duplicates": result.removed_duplicates,
            "removed_non_objects": result.removed_non_objects,
        },
        "scope_note": "The original dataset is never modified; cleaned rows are written to a distinct output path using an atomic replacement.",
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    console.print(Panel.fit(
        f"[bold cyan]AXIOM DATASET CLEANING[/bold cyan]\n\n"
        f"Source:             {result.source}\n"
        f"Output:             {result.output}\n\n"
        f"Input rows:         {result.total_lines:,}\n"
        f"Kept:               {result.kept:,}\n"
        f"Invalid removed:    {result.removed_invalid:,}\n"
        f"Empty removed:      {result.removed_empty:,}\n"
        f"Duplicates removed: {result.removed_duplicates:,}\n"
        f"Non-object rows:    {result.removed_non_objects:,}",
        title="AXIOM",
    ))

@mcp_app.command("serve")
def mcp_serve():
    """Run the AXIOM MCP server over stdio."""
    from axiom.mcp.server import mcp

    mcp.run()


if __name__ == "__main__":
    app()
