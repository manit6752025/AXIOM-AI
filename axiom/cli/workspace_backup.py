"""CLI commands for full workspace backup and isolated restore."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from axiom.cli.errors import emit_error
from axiom.core.diagnostics import discover_workspace_root
from axiom.core.workspace_backup import (
    MAX_BACKUP_ARCHIVE_BYTES,
    WorkspaceBackupError,
    create_workspace_backup,
    restore_workspace_backup,
)
from axiom.core.workspaces import WorkspaceCatalogError, WorkspaceRegistry
from axiom.path_safety import open_regular_file

backup_app = typer.Typer(
    help="Create and restore full workspace ZIP64 backups.",
    no_args_is_help=True,
)
console = Console()


def _workspace_root(workspace: Path | None) -> Path:
    try:
        return discover_workspace_root(workspace)
    except (OSError, ValueError) as exc:
        raise WorkspaceBackupError("Workspace directory could not be resolved safely.") from exc


def _validate_output_parent(destination: Path) -> tuple[Path, Path]:
    target = destination.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.abspath(target))
    parent = target.parent
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise WorkspaceBackupError("Backup output parent paths must not contain symbolic links.")
    if not parent.is_dir():
        raise WorkspaceBackupError("Backup output parent must be an existing directory.")
    if target.is_symlink() or target.exists():
        raise FileExistsError("Backup output already exists; choose another path.")
    return parent, target


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open_regular_file(path) as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@backup_app.command("create")
def backup_create(
    output: Path = typer.Option(
        Path("axiom-workspace-backup.zip"),
        "--output",
        "-o",
        help="New ZIP64 archive path. Existing files are never overwritten.",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        help="Workspace root; defaults to AXIOM workspace discovery.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable backup report."),
) -> None:
    """Create a full workspace backup with per-file SHA-256 checksums."""
    temporary: str | None = None
    try:
        root = _workspace_root(workspace)
        parent, destination = _validate_output_parent(output)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".backup-tmp",
            dir=parent,
        )
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w+b") as archive:
            report = create_workspace_backup(
                root,
                archive,
                exclude_paths=[destination, Path(temporary)],
            )
            archive.flush()
            os.fsync(archive.fileno())
            archive_bytes = archive.tell()
        if archive_bytes <= 0 or archive_bytes > MAX_BACKUP_ARCHIVE_BYTES:
            raise WorkspaceBackupError("Backup archive exceeds the 105 GiB compressed-size limit.")

        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise
        except (NotImplementedError, OSError) as link_error:
            # Some Windows/filesystem combinations do not support hard links.
            # Fall back to an exclusive, no-follow create and remove a partial
            # destination if copying fails.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            descriptor = os.open(destination, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as target:
                    with open(temporary, "rb") as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            except BaseException:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
                raise WorkspaceBackupError("Backup archive could not be committed safely.") from link_error
        os.unlink(temporary)
        temporary = None
        payload = {
            "schema_version": 1,
            "status": "created",
            "archive": str(destination),
            "archive_bytes": archive_bytes,
            "archive_sha256": _hash_file(destination),
            **report,
        }
    except FileExistsError as exc:
        emit_error(str(exc), code="workspace_backup_exists", exit_code=2)
    except (OSError, ValueError, WorkspaceBackupError, typer.BadParameter) as exc:
        emit_error(
            f"Workspace backup failed safely: {exc}",
            code="workspace_backup_failed",
            exit_code=1,
        )
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    console.print(Panel.fit(
        f"Archive: {payload['archive']}\n"
        f"Files: {payload['file_count']:,}\n"
        f"Workspace bytes: {payload['total_bytes']:,}\n"
        f"Archive bytes: {payload['archive_bytes']:,}\n"
        f"Manifest SHA-256: {payload['manifest_sha256']}\n"
        f"Archive SHA-256: {payload['archive_sha256']}\n"
        f"Skipped symlinks: {payload['skipped']['symlink_files'] + payload['skipped']['symlink_directories']:,}\n"
        "Includes workspace configuration, .axiom metadata, datasets and model weights. "
        "Store the archive as sensitive data.",
        title="AXIOM WORKSPACE BACKUP",
    ))


@backup_app.command("restore")
def backup_restore(
    archive: Path = typer.Argument(..., help="A previously created AXIOM workspace backup ZIP64 archive."),
    name: str = typer.Option(..., "--name", help="Name for the new restored workspace."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable restore report."),
) -> None:
    """Verify checksums and restore the archive into a new managed workspace."""
    staging: Path | None = None
    try:
        source = archive.expanduser()
        if source.is_symlink():
            raise WorkspaceBackupError("Backup archive must not be a symbolic link.")
        info = source.stat()
        if not source.is_file() or info.st_size <= 0 or info.st_size > MAX_BACKUP_ARCHIVE_BYTES:
            raise WorkspaceBackupError("Backup archive must be a bounded regular file.")
        registry = WorkspaceRegistry()
        staging = registry.create_restore_staging_directory()
        try:
            with open_regular_file(source) as stream:
                report = restore_workspace_backup(stream, staging)
            workspace = registry.create_from_staging_directory(name, staging)
            staging = None
        except BaseException:
            raise
        payload = {
            "schema_version": 1,
            "status": "restored",
            "workspace": workspace,
            "backup": report,
            "scope_note": "The archive was restored as a new workspace; no existing workspace was overwritten.",
        }
    except (OSError, ValueError, WorkspaceBackupError, WorkspaceCatalogError, typer.BadParameter) as exc:
        emit_error(
            f"Workspace restore failed safely: {exc}",
            code="workspace_backup_restore_failed",
            exit_code=1,
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    console.print(Panel.fit(
        f"Workspace: {payload['workspace']['name']}\n"
        f"Workspace ID: {payload['workspace']['id']}\n"
        f"Files restored: {payload['backup']['file_count']:,}\n"
        f"Bytes restored: {payload['backup']['total_bytes']:,}\n"
        f"Backup ID: {payload['backup']['backup_id']}\n"
        "Existing workspaces were not modified.",
        title="AXIOM WORKSPACE RESTORE",
    ))


def register(workspace_app: typer.Typer) -> None:
    """Attach backup commands to the existing axiom workspace command family."""
    workspace_app.add_typer(backup_app, name="backup")
