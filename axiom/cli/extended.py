"""Additional AXIOM CLI commands."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.daemon import local_summary
from axiom.core.diagnostics import diagnose_workspace
from axiom.cli.support_bundle import register as register_support_bundle_cli
from axiom.cli.roadmap import roadmap_app as roadmap_cli_app
from axiom.datasets.inspector import inspect_dataset
from axiom.version import __version__

AXIOM_VERSION = __version__
console = Console()

PROJECT_PATHS = (
    "axiom.yaml",
    "README.md",
    "data",
    "models",
    "experiments",
    "evaluations",
    "outputs",
)


def dataset_inspection_payload(result) -> dict[str, object]:
    """Serialize dataset inspection results consistently across CLI commands."""
    return {
        "schema_version": 1,
        "dataset": {
            "path": str(result.path),
            "format": result.format,
            "samples": result.samples,
            "valid": result.valid,
            "invalid": result.invalid,
            "empty": result.empty,
            "duplicates": result.duplicates,
            "estimated_tokens": result.estimated_tokens,
            "fields": list(result.fields),
            "errors": [{"line": number, "message": message} for number, message in result.errors],
            "errors_limit": 50,
        },
        "scope_note": "Estimated tokens are a rough serialized-character estimate, not a tokenizer measurement.",
    }


def package_version() -> str:
    for name in ("axiom-all", "axiom-ai"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "development"


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except OSError:
        return ""


def human_size(size: int) -> str:
    value = float(size)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{size} B"


def project_detected(root: Path) -> bool:
    """Return whether the current directory has an AXIOM config file."""
    return (root / "axiom.yaml").is_file()


def missing_project_paths(root: Path) -> list[str]:
    """Return standard project paths that are absent from ``root``."""
    return [path for path in PROJECT_PATHS if not (root / path).exists()]


def next_commands(root: Path) -> tuple[str, ...]:
    """Choose safe, local next commands for the current directory state."""
    if not project_detected(root):
        return (
            "axiom init my-ai",
            "cd my-ai",
            "axiom guide",
        )

    if missing_project_paths(root):
        return (
            "axiom config validate",
            "axiom project validate",
            "axiom status",
        )

    return (
        "axiom config validate",
        "axiom status",
        "axiom model list",
    )


def print_next_commands(commands: tuple[str, ...]) -> None:
    console.print("Next commands:")

    for index, command in enumerate(commands, start=1):
        console.print(f"  {index}. [cyan]{command}[/cyan]")


def register(app, model_app, dataset_app) -> None:

    @app.command("doctor")
    def doctor(
        json_output: bool = typer.Option(False, "--json", help="Emit stable machine-readable diagnostics."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace directory to inspect."),
        strict: bool = typer.Option(False, "--strict", help="Treat workspace warnings as failures, including the command exit status in JSON mode."),
    ) -> None:
        """Inspect the current workspace, Python dependencies, and common tools."""
        try:
            report = diagnose_workspace(workspace)
        except (OSError, ValueError):
            message = {"schema_version": 1, "status": "error", "message": "Workspace could not be resolved safely."}
            typer.echo(json.dumps(message, sort_keys=True) if json_output else "Error: workspace could not be resolved safely.", err=not json_output)
            raise typer.Exit(code=2)
        tool_checks = [
            {"name": name, "available": bool(shutil.which(command)), "path": shutil.which(command)}
            for name, command in (("Python", "python3"), ("Python fallback", "python"), ("Git", "git"), ("Node.js", "node"), ("npm", "npm"))
        ]
        payload = {"schema_version": 1, "version": AXIOM_VERSION, "workspace": report, "tools": tool_checks}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            if report["summary"]["failed"] or (strict and report["summary"]["warnings"]):
                raise typer.Exit(code=1)
            return

        typer.echo(f"AXIOM Doctor — {AXIOM_VERSION}")
        typer.echo(f"Workspace: {report['workspace']['name']}")
        for check in report["checks"]:
            marker = "OK" if check["status"] == "pass" else "FAIL" if check["status"] == "fail" else "WARN" if check["status"] == "warn" else "INFO"
            typer.echo(f"[{marker:<4}] {check['title']}: {check['detail']}")
            if check["remediation"]:
                typer.echo(f"       Next: {check['remediation']}")
        for item in tool_checks:
            marker = "OK" if item["available"] else "WARN"
            typer.echo(f"[{marker:<4}] Tool {item['name']}: {item['path'] or 'not found'}")
        summary = report["summary"]
        typer.echo(f"Doctor complete: {summary['passed']} passed, {summary['warnings']} warning(s), {summary['failed']} failed")
        if strict and summary["warnings"] and not summary["failed"]:
            typer.echo("Strict mode: workspace warnings are treated as failures.")
        if summary["failed"] or (strict and summary["warnings"]):
            raise typer.Exit(code=1)

    @app.command("info")
    def info() -> None:
        """Show local AXIOM and Python environment information."""
        cwd = Path.cwd()

        console.print(
            Panel.fit(
                f"[bold cyan]AXIOM[/bold cyan] v{AXIOM_VERSION}\n\n"
                f"Package: {package_version()}\n"
                f"Python: {platform.python_version()}\n"
                f"Platform: {platform.platform()}\n"
                f"Executable: {sys.executable}\n"
                f"Working directory: {cwd}\n"
                f"Project config: "
                f"{'found' if (cwd / 'axiom.yaml').exists() else 'not found'}\n"
                f"Git branch: "
                f"{git_output('branch', '--show-current') or 'n/a'}",
                title="AXIOM INFO",
            )
        )

    @app.command("status")
    def status(
        json_output: bool = typer.Option(False, "--json", help="Emit stable machine-readable status."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace directory to inspect."),
    ) -> None:
        """Show workspace health and local Git status without running jobs."""
        try:
            report = diagnose_workspace(workspace)
        except (OSError, ValueError):
            message = {"schema_version": 1, "status": "error", "message": "Workspace could not be resolved safely."}
            typer.echo(json.dumps(message, sort_keys=True) if json_output else "Error: workspace could not be resolved safely.", err=not json_output)
            raise typer.Exit(code=2)
        dirty = bool(git_output("status", "--porcelain"))
        payload = {
            "schema_version": 1,
            "version": AXIOM_VERSION,
            "workspace": report["workspace"],
            "health": {"status": report["status"], "summary": report["summary"]},
            "git": {
                "branch": git_output("branch", "--show-current") or None,
                "commit": git_output("rev-parse", "--short", "HEAD") or None,
                "dirty": dirty,
            },
        }
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            return

        table = Table(title="AXIOM STATUS")
        table.add_column("Item")
        table.add_column("Value")
        table.add_row("Version", AXIOM_VERSION)
        table.add_row("Workspace", report["workspace"]["name"])
        table.add_row("Project config", "found" if report["workspace"]["project_config_present"] else "not found")
        table.add_row("Workspace health", report["status"].upper())
        table.add_row("Checks", f"{report['summary']['passed']} passed / {report['summary']['warnings']} warnings / {report['summary']['failed']} failed")
        table.add_row("Git branch", payload["git"]["branch"] or "n/a")
        table.add_row("Git commit", payload["git"]["commit"] or "n/a")
        table.add_row("Working tree", "modified" if dirty else "clean")
        console.print(table)

    @app.command("summary")
    def summary(
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Print a stable machine-readable local snapshot.",
        ),
    ) -> None:
        """Summarize the current project, machine, tools, and next command."""

        payload = local_summary()
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            return

        project = payload["project"]
        hardware = payload["hardware"]
        tools = payload["tools"]
        console.print(
            Panel.fit(
                f"Directory: {payload['working_directory']}\n"
                f"Project: {'detected' if project['detected'] else 'not detected'}\n"
                f"Missing paths: {', '.join(project['missing_paths']) or 'none'}\n\n"
                f"System: {hardware['os_name']} / {hardware['architecture']}\n"
                f"CPU: {hardware['cpu_cores']} cores, RAM: {hardware['ram_gb']:.2f} GB\n"
                f"GPU: {hardware['gpu_name'] or 'not detected'}\n\n"
                f"Tools on PATH: {tools['available']} of {tools['total']}\n"
                f"Missing tools: {', '.join(tools['missing']) or 'none'}",
                title="AXIOM SUMMARY",
            )
        )
        console.print(
            f"Next: [cyan]{payload['recommended']['command']}[/cyan]\n"
            f"{payload['recommended']['reason']}"
        )

    @app.command("guide")
    def guide() -> None:
        """Explain AXIOM and suggest the next local commands."""
        root = Path.cwd()
        detected = project_detected(root)

        if detected:
            missing = missing_project_paths(root)
            project_line = "detected (axiom.yaml found)"
            structure_line = (
                "standard project paths are present"
                if not missing
                else f"missing: {', '.join(missing)}"
            )
        else:
            project_line = "not detected (axiom.yaml is missing)"
            structure_line = "create a project with the commands below"

        console.print(
            Panel.fit(
                "[bold cyan]AXIOM GUIDE[/bold cyan]\n\n"
                "AXIOM is a local-first command-line tool for organizing AI "
                "model and dataset work. It can inspect local files, check "
                "project structure, keep local model metadata, and make "
                "hardware-aware training-plan estimates. Some commands can "
                "use external services; this guide only checks the current "
                "directory.\n\n"
                f"Current directory: {root}\n"
                f"AXIOM project: {project_line}\n"
                f"Project structure: {structure_line}",
                title="AXIOM",
            )
        )
        print_next_commands(next_commands(root))

    @app.command("check")
    def check() -> None:
        """Quickly check for a local AXIOM project and its standard paths."""
        root = Path.cwd()

        if not project_detected(root):
            console.print(f"[yellow]No AXIOM project detected:[/yellow] {root}")
            console.print("Next: [cyan]axiom init my-ai[/cyan]")
            return

        missing = missing_project_paths(root)
        console.print(f"[green]✓ AXIOM project detected:[/green] {root}")

        if missing:
            console.print(
                f"[yellow]Missing standard paths:[/yellow] {', '.join(missing)}"
            )
        else:
            console.print("[green]✓ Standard project paths are present.[/green]")

        console.print("Next: [cyan]axiom config validate[/cyan]")



    @model_app.command("search")
    def model_search(
        query: str = typer.Argument(..., help="Text to match in local model paths"),
        path: str = typer.Option("models", "--path", help="Directory to search for matching local model paths"),
        json_output: bool = typer.Option(False, "--json", help="Emit search matches as JSON."),
    ) -> None:
        """Search local model paths by name."""
        root = Path(path)
        matches = [
            item for item in root.rglob("*")
            if root.is_dir() and query.lower() in str(item).lower()
        ] if root.is_dir() else []
        matches = sorted(matches, key=lambda item: str(item).casefold())
        records = []
        for item in matches:
            if item.is_symlink():
                continue
            is_file = item.is_file()
            records.append({
                "path": str(item),
                "kind": "file" if is_file else "directory",
                "size_bytes": item.stat().st_size if is_file else None,
            })

        if json_output:
            typer.echo(json.dumps({
                "schema_version": 1,
                "query": query,
                "root": str(root),
                "total": len(records),
                "matches": records,
            }, ensure_ascii=False, sort_keys=True))
            return
        if not root.exists() or not root.is_dir():
            console.print(f"[yellow]No model directory:[/yellow] {root}")
            return
        if not records:
            console.print("[yellow]No matches found.[/yellow]")
            return

        table = Table(title="AXIOM MODEL SEARCH")
        table.add_column("Path")
        table.add_column("Type")
        table.add_column("Size")
        for item in records:
            table.add_row(
                item["path"], item["kind"],
                human_size(item["size_bytes"]) if item["size_bytes"] is not None else "-",
            )
        console.print(table)

    @dataset_app.command("validate")
    def dataset_validate(
        path: str = typer.Argument(..., help="Local JSONL dataset file"),
        json_output: bool = typer.Option(False, "--json", help="Emit complete validation results as JSON."),
    ) -> None:
        """Check JSONL records and report duplicate objects."""
        try:
            result = inspect_dataset(path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            if json_output:
                typer.echo(json.dumps({
                    "schema_version": 1,
                    "error": {"code": "dataset_validation_failed", "message": str(exc)},
                }, ensure_ascii=False, sort_keys=True))
            else:
                console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        if json_output:
            typer.echo(json.dumps(dataset_inspection_payload(result), ensure_ascii=False, sort_keys=True))
            if result.invalid:
                raise typer.Exit(code=1)
            return

        table = Table(title="AXIOM DATASET VALIDATION")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Records", str(result.samples))
        table.add_row("Valid", str(result.valid))
        table.add_row("Invalid", str(result.invalid))
        table.add_row("Empty", str(result.empty))
        table.add_row("Duplicates", str(result.duplicates))
        console.print(table)
        for line_number, message in result.errors:
            console.print(f"[red]INVALID[/red] line {line_number}: {message}")
        if result.invalid:
            raise typer.Exit(code=1)

    @dataset_app.command("stats")
    def dataset_stats(
        path: str = typer.Argument(..., help="Local JSONL dataset file"),
        json_output: bool = typer.Option(False, "--json", help="Emit dataset statistics as JSON."),
    ) -> None:
        """Show local JSONL counts and a rough token estimate."""
        try:
            result = inspect_dataset(path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            if json_output:
                typer.echo(json.dumps({
                    "schema_version": 1,
                    "error": {"code": "dataset_statistics_failed", "message": str(exc)},
                }, ensure_ascii=False, sort_keys=True))
            else:
                console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        if json_output:
            typer.echo(json.dumps(dataset_inspection_payload(result), ensure_ascii=False, sort_keys=True))
            return

        console.print(Panel.fit(
            f"[bold cyan]AXIOM DATASET STATS[/bold cyan]\n\n"
            f"Records: {result.valid:,}\n"
            f"Invalid: {result.invalid:,}\n"
            f"Empty: {result.empty:,}\n"
            f"Duplicates: {result.duplicates:,}\n"
            f"Fields: {', '.join(result.fields) or 'None'}\n"
            f"Estimated tokens: {result.estimated_tokens:,}",
            title="AXIOM",
        ))

    project_app = typer.Typer(
        help="Inspect and validate the current local AXIOM project."
    )
    app.add_typer(project_app, name="project")

    @project_app.command("info")
    def project_info() -> None:
        """Show local project markers in the current directory."""
        root = Path.cwd()

        console.print(
            Panel.fit(
                f"[bold cyan]AXIOM PROJECT[/bold cyan]\n\n"
                f"Path: {root}\n"
                f"Config: "
                f"{'found' if (root / 'axiom.yaml').exists() else 'missing'}\n"
                f"README: "
                f"{'found' if (root / 'README.md').exists() else 'missing'}",
                title="AXIOM",
            )
        )

    @project_app.command("validate")
    def project_validate() -> None:
        """Check expected AXIOM project paths in the current directory."""
        root = Path.cwd()

        required = [
            "axiom.yaml",
            "README.md",
            "data",
            "models",
            "experiments",
            "evaluations",
            "outputs",
        ]

        missing = [item for item in required if not (root / item).exists()]

        if missing:
            console.print("[red]Project validation failed.[/red]")

            for item in missing:
                console.print(f"  [red]×[/red] {item}")

            raise typer.Exit(code=1)

        console.print("[green]✓ AXIOM project structure is valid.[/green]")

    config_app = typer.Typer(help="Inspect local AXIOM project configuration.")
    app.add_typer(config_app, name="config")

    @config_app.command("show")
    def config_show(
        path: str = typer.Option(
            "axiom.yaml",
            "--path",
            help="Path to a local axiom.yaml file",
        ),
    ) -> None:
        """Print a local axiom.yaml file."""
        config = Path(path)

        if not config.is_file():
            console.print(f"[red]Error:[/red] config not found: {config}")
            raise typer.Exit(code=1)

        typer.echo(config.read_text(encoding="utf-8"))

    @config_app.command("validate")
    def config_validate(
        path: str = typer.Option("axiom.yaml", "--path", help="Path to a local axiom.yaml file"),
        json_output: bool = typer.Option(False, "--json", help="Emit a stable machine-readable result."),
    ) -> None:
        """Validate a local YAML mapping and suggest safe, non-mutating repairs."""
        config = Path(path)
        status = "invalid"
        valid = False
        error_code: str | None = None
        suggestions: list[str] = []

        if config.is_symlink():
            error_code = "config_symlink_forbidden"
            message = "Configuration path must not be a symbolic link."
            suggestions = [
                "Choose a regular local file path.",
                "Review the symlink target yourself before copying content; AXIOM will not follow or replace it.",
            ]
        elif not config.is_file():
            error_code = "config_not_found"
            message = "Configuration file was not found."
            suggestions = [
                "Pass an existing configuration file with --path.",
                "Create a starter project with axiom init <project-name> if you need a new axiom.yaml.",
            ]
        else:
            try:
                config_size = config.stat().st_size
            except OSError:
                error_code = "config_unreadable"
                message = "Configuration metadata could not be read."
                suggestions = [
                    "Check that the file exists and your account has permission to read it.",
                    "Choose a regular local file that is at most 256 KiB.",
                ]
            else:
                if config_size > 256 * 1024:
                    error_code = "config_too_large"
                    message = "Configuration exceeds the 256 KiB validation limit."
                    suggestions = [
                        "Review the file and remove unrelated or generated content before validating again.",
                        "Keep only the sections AXIOM needs; AXIOM will not rewrite the file automatically.",
                    ]
                else:
                    try:
                        contents = config.read_text(encoding="utf-8")
                        parsed = yaml.safe_load(contents)
                    except (OSError, UnicodeError, yaml.YAMLError):
                        error_code = "config_invalid"
                        message = "invalid axiom.yaml: configuration could not be parsed as bounded UTF-8 YAML; file values are withheld."
                        suggestions = [
                            "Check YAML indentation, colons, brackets, and quoting in your editor.",
                            "Ensure the document is UTF-8 and within the 256 KiB limit.",
                            "Re-run axiom config validate --path <file> after editing; AXIOM will not edit it for you.",
                        ]
                    else:
                        if not contents.strip() or not isinstance(parsed, dict):
                            error_code = "config_not_mapping"
                            message = "Configuration must be a non-empty YAML mapping."
                            suggestions = [
                                "Use YAML key/value pairs at the top level rather than a list, scalar, or empty document.",
                                "Put settings under mapping sections such as project, model, dataset, training, evaluation, or deployment.",
                            ]
                        else:
                            known_sections = ("project", "model", "dataset", "training", "evaluation", "deployment")
                            bad_sections = [
                                key for key in known_sections
                                if key in parsed and parsed[key] is not None and not isinstance(parsed[key], dict)
                            ]
                            if bad_sections:
                                error_code = "config_section_not_mapping"
                                message = "Known configuration sections must be YAML mappings."
                                suggestions = [
                                    "Convert each incorrectly shaped section into a YAML key/value mapping.",
                                    "Validate again after editing; section contents are never printed or rewritten by this command.",
                                ]
                            else:
                                valid = True
                                status = "valid"
                                message = "Configuration is valid YAML and known sections have compatible types."

        result = {
            "schema_version": 1,
            "status": status,
            "valid": valid,
            "path": str(config),
            "message": message,
            "error_code": error_code,
            "suggestions": suggestions,
            "repair_applied": False,
            "repair_mode": "suggestions_only",
        }
        if json_output:
            typer.echo(json.dumps(result, indent=2, sort_keys=True))
        elif valid:
            console.print(f"[green]✓ {message}[/green]")
        else:
            console.print(f"[red]Error:[/red] {message}")
            if suggestions:
                console.print("[bold]Suggested next steps:[/bold]")
                for suggestion in suggestions:
                    console.print(f"  • {suggestion}")
        if not valid:
            raise typer.Exit(code=1)

    app.add_typer(roadmap_cli_app, name="roadmap")
    register_support_bundle_cli(app)
