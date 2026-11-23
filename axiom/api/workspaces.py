"""Workspace discovery, creation and naming endpoints."""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Request
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.workspace_context import active_workspace_id
from axiom.core.workspaces import WorkspaceCatalogError, WorkspaceNotFoundError, WorkspaceRegistry
from axiom.core.workspace_backup import (
    MAX_BACKUP_ARCHIVE_BYTES,
    WorkspaceBackupError,
    create_workspace_backup,
    restore_workspace_backup,
)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


class WorkspaceNameInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=64)


@router.get("")
def list_workspaces() -> dict:
    manager = WorkspaceRegistry()
    try:
        workspaces = manager.list_workspaces()
        active_id = active_workspace_id() or manager.default_workspace_id()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Workspace catalog is unavailable.") from exc
    return {
        "schema_version": 1,
        "active_workspace_id": active_id,
        "default_workspace_id": manager.default_workspace_id(),
        "workspaces": workspaces,
        "scope_note": "Workspace filesystem paths remain local and are not returned by this API.",
    }


@router.post("", status_code=201)
def create_workspace(request: WorkspaceNameInput) -> dict:
    try:
        workspace = WorkspaceRegistry().create(request.name)
    except WorkspaceCatalogError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Workspace could not be created safely.") from exc
    return {"workspace": workspace}


@router.get("/{workspace_id}/backup")
def download_workspace_backup(workspace_id: str) -> StreamingResponse:
    """Stream a checksum-verified backup of one registered workspace."""
    try:
        workspace = WorkspaceRegistry().resolve(workspace_id)
        stream = tempfile.TemporaryFile(mode="w+b")
        report = create_workspace_backup(workspace.root, stream)
        stream.seek(0)
    except WorkspaceNotFoundError as exc:
        if "stream" in locals():
            stream.close()
        raise HTTPException(status_code=404, detail="Workspace not found.") from exc
    except (WorkspaceCatalogError, WorkspaceBackupError, OSError, ValueError) as exc:
        if "stream" in locals():
            stream.close()
        raise HTTPException(status_code=422, detail="Workspace backup could not be created safely.") from exc

    def chunks() -> Iterator[bytes]:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            yield block

    return StreamingResponse(
        chunks(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="axiom-workspace-backup.zip"',
            "Content-Length": str(report["archive_bytes"]),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Axiom-Backup-Files": str(report["file_count"]),
            "X-Axiom-Backup-Manifest-SHA256": report["manifest_sha256"],
        },
        background=BackgroundTask(stream.close),
    )


@router.post("/backup/restore", status_code=201)
async def restore_workspace_backup_endpoint(
    request: Request,
    name: str = Query(..., min_length=1, max_length=64),
) -> dict:
    """Verify an uploaded archive and publish it as a new managed workspace."""
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in {"application/zip", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload a ZIP workspace backup archive.")
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_BACKUP_ARCHIVE_BYTES:
        raise HTTPException(status_code=413, detail="Workspace backup exceeds the 105 GiB upload limit.")

    manager = WorkspaceRegistry()
    staging: Path | None = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as upload:
            total = 0
            async for block in request.stream():
                total += len(block)
                if total > MAX_BACKUP_ARCHIVE_BYTES:
                    raise HTTPException(status_code=413, detail="Workspace backup exceeds the 105 GiB upload limit.")
                upload.write(block)
            upload.flush()
            upload.seek(0)
            staging = manager.create_restore_staging_directory()
            report = restore_workspace_backup(upload, staging)
        workspace = manager.create_from_staging_directory(name, staging)
        staging = None
    except HTTPException:
        raise
    except WorkspaceBackupError as exc:
        raise HTTPException(status_code=422, detail="Workspace backup was rejected; no workspace was registered.") from exc
    except WorkspaceCatalogError as exc:
        raise HTTPException(status_code=409, detail="A new workspace could not be created from this backup.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail="Workspace restore could not be completed safely.") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "schema_version": 1,
        "workspace": workspace,
        "backup": report,
        "scope_note": "The backup was restored into a new managed workspace. The active workspace was not modified.",
    }


@router.patch("/{workspace_id}")
def rename_workspace(workspace_id: str, request: WorkspaceNameInput) -> dict:
    try:
        workspace = WorkspaceRegistry().rename(workspace_id, request.name)
    except WorkspaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found.") from exc
    except WorkspaceCatalogError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Workspace could not be renamed.") from exc
    return {"workspace": workspace}
