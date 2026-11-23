"""CLI entry points for safe workspace metadata transfer."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from axiom.core.diagnostics import discover_workspace_root
from axiom.core.workspace_metadata import (
    MAX_ARCHIVE_BYTES,
    export_workspace_metadata,
    import_workspace_metadata,
)

workspace_app = typer.Typer(help="Transfer allowlisted workspace metadata.", no_args_is_help=True)
metadata_app = typer.Typer(help="Export or import workspace metadata archives.", no_args_is_help=True)
console = Console()


def _resolve_root(workspace: Path | None) -> Path:
    try:
        return discover_workspace_root(workspace)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter("Workspace directory could not be resolved safely.") from exc


def _write_archive(path: Path, archive: bytes, *, force: bool) -> Path:
    destination = path.expanduser()
    if destination.exists() and destination.is_dir():
        raise ValueError("Archive output path is a directory.")
    if destination.is_symlink():
        raise ValueError("Archive output path must not be a symbolic link.")
    if destination.exists() and not force:
        raise FileExistsError("Archive output already exists; pass --force to replace it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(archive)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.is_symlink():
            raise ValueError("Archive output path became a symbolic link.")
        if destination.exists() and not force:
            raise FileExistsError("Archive output already exists; pass --force to replace it.")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return destination.resolve(strict=True)


@metadata_app.command("export")
def export_metadata(
    output: Path = typer.Option(
        Path("axiom-workspace-metadata.zip"),
        "--output",
        "-o",
        help="ZIP archive destination.",
    ),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace root; defaults to AXIOM workspace discovery."),
    force: bool = typer.Option(False, "--force", help="Replace an existing archive destination."),
    json_output: bool = typer.Option(False, "--json", help="Emit the export summary as JSON."),
) -> None:
    """Export allowlisted registries, workflows, experiments, evaluations, and agent profiles."""
    try:
        root = _resolve_root(workspace)
        archive = export_workspace_metadata(root)
        destination = _write_archive(output, archive, force=force)
    except FileExistsError as exc:
        console.print(f"[red]Error:[/red] {exc}", stderr=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, typer.BadParameter) as exc:
        console.print(f"[red]Error:[/red] Workspace metadata export failed safely: {exc}", stderr=True)
        raise typer.Exit(code=1) from exc

    from zipfile import ZipFile
    import io

    with ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    payload = {
        "schema_version": 1,
        "archive": str(destination),
        "size_bytes": len(archive),
        "collections": {
            name: descriptor["records"]
            for name, descriptor in manifest["collections"].items()
        },
        "excluded": manifest["excluded"],
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    console.print(Panel.fit(
        f"Archive: {destination}\n"
        f"Size: {len(archive):,} bytes\n"
        f"Collections: {', '.join(f'{name} ({count})' for name, count in payload['collections'].items())}\n"
        "Excluded: credentials, MCP grants, integration endpoints, jobs, datasets, and model weights",
        title="AXIOM WORKSPACE METADATA EXPORT",
    ))


@metadata_app.command("import")
def import_metadata(
    archive: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="AXIOM workspace metadata ZIP archive."),
    workspace: Path | None = typer.Option(None, "--workspace", help="Destination workspace; defaults to AXIOM workspace discovery."),
    replace: bool = typer.Option(False, "--replace", help="Replace only the collections present in the archive instead of merging."),
    json_output: bool = typer.Option(False, "--json", help="Emit the import summary as JSON."),
) -> None:
    """Validate and merge a metadata archive without replacing unrelated workspace files."""
    try:
        root = _resolve_root(workspace)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Workspace metadata archive exceeds the 24 MiB limit.")
        result = import_workspace_metadata(archive.read_bytes(), root, replace=replace)
    except (OSError, ValueError, typer.BadParameter) as exc:
        payload = {"schema_version": 1, "error": {"code": "workspace_metadata_import_failed", "message": str(exc)}}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            console.print(f"[red]Error:[/red] Workspace metadata import failed: {exc}", stderr=True)
        raise typer.Exit(code=1) from exc

    payload = {"schema_version": 1, **result}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    counts = ", ".join(f"{name} ({count})" for name, count in result["imported"].items())
    console.print(Panel.fit(
        f"Workspace: {root}\nMode: {result['mode']}\nImported: {counts}\n"
        "No datasets, model weights, credentials, or MCP grants were imported.",
        title="AXIOM WORKSPACE METADATA IMPORT",
    ))


workspace_app.add_typer(metadata_app, name="metadata")


def register(app: typer.Typer) -> None:
    """Attach the workspace metadata command family to the root CLI."""
    app.add_typer(workspace_app, name="workspace")
