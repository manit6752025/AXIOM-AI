"""CLI commands for local, versioned Prompt Studio templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from axiom.cli.errors import emit_error
from axiom.core.prompt_templates import (
    MAX_BODY_CHARS,
    PromptTemplateError,
    PromptTemplateStore,
)

console = Console()


def _fail(exc: Exception) -> None:
    emit_error(str(exc), code="prompt_template_error")


def _read_body(body: str | None, body_file: Path | None) -> str:
    if body is not None and body_file is not None:
        raise typer.BadParameter("Use --body or --body-file, not both.")
    if body is not None:
        return body
    if body_file is None:
        raise typer.BadParameter("Provide --body or --body-file.")
    try:
        if body_file.is_symlink() or not body_file.is_file() or body_file.stat().st_size > MAX_BODY_CHARS * 4:
            raise typer.BadParameter("--body-file must be a regular, non-symlink file within the size limit.")
        return body_file.read_text(encoding="utf-8")
    except UnicodeError:
        raise typer.BadParameter("--body-file must contain UTF-8 text.") from None
    except OSError:
        raise typer.BadParameter("--body-file could not be read safely.") from None


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise typer.BadParameter(f"{label} must be valid JSON.") from None
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{label} must be a JSON object.")
    return parsed


def _json_variables(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise typer.BadParameter("--variables-json must be valid JSON.") from None
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise typer.BadParameter("--variables-json must be a JSON array of variable objects.")
    return parsed


def register(app: typer.Typer) -> None:
    command = typer.Typer(
        help="Create, validate, render and version local prompt templates. Rendering is text-only and never executes a model or tool.",
        no_args_is_help=True,
    )
    app.add_typer(command, name="prompt")

    @command.command("list")
    def list_templates(
        search: str = typer.Option("", "--search", help="Search template name, description and body."),
        limit: int = typer.Option(50, "--limit", min=1, max=100),
        offset: int = typer.Option(0, "--offset", min=0, max=1_000_000),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """List saved workspace-local prompt templates."""
        try:
            report = PromptTemplateStore().list_page(limit=limit, offset=offset, search=search)
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if json_output:
            typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
            return
        table = Table(title=f"PROMPT TEMPLATES · {report['total']}")
        table.add_column("Name", style="cyan")
        table.add_column("Version", justify="right")
        table.add_column("Variables", justify="right")
        table.add_column("Updated")
        for item in report["templates"]:
            table.add_row(item["name"], str(item["version"]), str(item["variable_count"]), item["updated_at"])
        console.print(table)

    @command.command("show")
    def show_template(
        template_id: str = typer.Argument(..., help="Prompt template ID."),
        json_output: bool = typer.Option(False, "--json", help="Emit the full template as JSON."),
    ) -> None:
        """Show a template body and its typed variable definitions."""
        try:
            record = PromptTemplateStore().get(template_id)
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if record is None:
            _fail(PromptTemplateError("Prompt template was not found."))
        if json_output:
            typer.echo(json.dumps(record, ensure_ascii=False, indent=2))
            return
        console.print(f"[bold]{record['name']}[/bold] · version {record['version']} · {record['id']}")
        console.print(record["description"])
        console.print(record["body"])
        console.print_json(json.dumps(record["variables"], ensure_ascii=False))

    @command.command("create")
    def create_template(
        name: str = typer.Argument(..., help="Unique template name."),
        body: str | None = typer.Option(None, "--body", help="Template text; alternatively use --body-file."),
        body_file: Path | None = typer.Option(None, "--body-file", exists=True, dir_okay=False, readable=True),
        description: str = typer.Option("", "--description", max_length=240),
        variables_json: str = typer.Option("[]", "--variables-json", help="JSON array of typed variable definitions."),
        json_output: bool = typer.Option(False, "--json", help="Emit the created template as JSON."),
    ) -> None:
        """Create a version-1 workspace-local template."""
        try:
            result = PromptTemplateStore().create(
                name=name, description=description, body=_read_body(body, body_file),
                variables=_json_variables(variables_json),
            )
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            console.print(f"[green]Created[/green] {result['name']} · version {result['version']} · {result['id']}")

    @command.command("update")
    def update_template(
        template_id: str = typer.Argument(..., help="Prompt template ID."),
        expected_version: int = typer.Option(..., "--expected-version", min=1, max=1_000_000),
        name: str = typer.Option(..., "--name", max_length=80),
        body: str | None = typer.Option(None, "--body"),
        body_file: Path | None = typer.Option(None, "--body-file", exists=True, dir_okay=False, readable=True),
        description: str = typer.Option("", "--description", max_length=240),
        variables_json: str = typer.Option("[]", "--variables-json"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Update a template only when its expected version still matches."""
        try:
            result = PromptTemplateStore().update(
                template_id, expected_version=expected_version, name=name,
                description=description, body=_read_body(body, body_file),
                variables=_json_variables(variables_json),
            )
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if result is None:
            _fail(PromptTemplateError("Prompt template was not found."))
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            console.print(f"[green]Updated[/green] {result['name']} · version {result['version']}")

    @command.command("versions")
    def list_versions(
        template_id: str = typer.Argument(...),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """List retained template revisions."""
        try:
            versions = PromptTemplateStore().versions(template_id)
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if versions is None:
            _fail(PromptTemplateError("Prompt template was not found."))
        if json_output:
            typer.echo(json.dumps({"template_id": template_id, "versions": versions}, ensure_ascii=False, indent=2))
            return
        table = Table(title="PROMPT TEMPLATE REVISIONS")
        table.add_column("Version", justify="right")
        table.add_column("Name")
        table.add_column("Updated")
        table.add_column("Current")
        for item in versions:
            table.add_row(str(item["version"]), item["name"], item["updated_at"], "yes" if item["current"] else "")
        console.print(table)

    @command.command("render")
    def render_template(
        template_id: str = typer.Argument(...),
        values_json: str = typer.Option("{}", "--values-json", help="Typed JSON scalar values keyed by variable name."),
        revision: int | None = typer.Option(None, "--revision", min=1, max=1_000_000),
        json_output: bool = typer.Option(False, "--json", help="Emit render metadata and output as JSON."),
    ) -> None:
        """Render a template using literal typed substitution; no model or tool is invoked."""
        try:
            result = PromptTemplateStore().render(template_id, _json_object(values_json, "--values-json"), version=revision)
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if result is None:
            _fail(PromptTemplateError("Prompt template was not found."))
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            typer.echo(result["rendered"], nl=False)

    @command.command("delete")
    def delete_template(
        template_id: str = typer.Argument(...),
        expected_version: int = typer.Option(..., "--expected-version", min=1, max=1_000_000),
        yes: bool = typer.Option(False, "--yes", help="Confirm deletion non-interactively."),
    ) -> None:
        """Delete a template after an explicit version check and --yes."""
        if not yes:
            raise typer.Exit(code=2)
        try:
            deleted = PromptTemplateStore().delete(template_id, expected_version=expected_version)
        except (PromptTemplateError, OSError) as exc:
            _fail(exc)
        if not deleted:
            _fail(PromptTemplateError("Prompt template was not found."))
        console.print(f"[green]Deleted[/green] {template_id}")
