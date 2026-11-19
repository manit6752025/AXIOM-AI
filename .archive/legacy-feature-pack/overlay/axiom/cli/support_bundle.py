"""CLI command for exporting a privacy-preserving support bundle."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import typer

from axiom.core.diagnostics import create_support_bundle


def register(app: typer.Typer) -> None:
    @app.command("support-bundle")
    def support_bundle(
        output: Path = typer.Option(
            Path("axiom-support-bundle.zip"),
            "--output",
            "-o",
            help="Output ZIP path. Existing files are not overwritten without --force.",
        ),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            help="Workspace directory to inspect; defaults to AXIOM_WORKSPACE or discovered root.",
        ),
        force: bool = typer.Option(False, "--force", help="Replace an existing regular output file."),
        json_output: bool = typer.Option(False, "--json", help="Print a machine-readable result."),
    ) -> None:
        """Export privacy-safe diagnostics for support; never include data files."""
        destination = output.expanduser()
        if destination.is_symlink():
            typer.echo("Error: output path must not be a symbolic link.", err=True)
            raise typer.Exit(code=2)
        if destination.exists() and not destination.is_file():
            typer.echo("Error: output path must be a regular file.", err=True)
            raise typer.Exit(code=2)
        if destination.exists() and not force:
            typer.echo("Error: output file already exists; pass --force to replace it.", err=True)
            raise typer.Exit(code=2)
        parent = destination.parent
        if parent.is_symlink() or not parent.is_dir():
            typer.echo("Error: output parent must be an existing, non-symlink directory.", err=True)
            raise typer.Exit(code=2)
        try:
            archive = create_support_bundle(workspace)
            flags = os.O_WRONLY | os.O_CREAT
            flags |= os.O_TRUNC if force else os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(destination, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(archive)
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    # Preserve the original write error if descriptor cleanup also fails.
                    pass
                raise
        except FileExistsError:
            typer.echo("Error: output file already exists; pass --force to replace it.", err=True)
            raise typer.Exit(code=2) from None
        except (OSError, ValueError) as exc:
            typer.echo(f"Error: could not write support bundle ({type(exc).__name__}).", err=True)
            raise typer.Exit(code=1) from None

        payload = {
            "schema_version": 1,
            "status": "created",
            "output": str(destination),
            "bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(),
            "privacy": "diagnostics and environment metadata only; no config values, environment variables, datasets, model files, or logs",
        }
        if json_output:
            import json
            typer.echo(json.dumps(payload, sort_keys=True))
        else:
            typer.echo(f"Support bundle created: {destination} ({len(archive)} bytes)")
            typer.echo(f"SHA-256: {payload['sha256']}")
            typer.echo("Review the archive before sharing it.")
