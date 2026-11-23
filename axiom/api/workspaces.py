"""Workspace discovery, creation and naming endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.workspace_context import active_workspace_id
from axiom.core.workspaces import WorkspaceCatalogError, WorkspaceNotFoundError, WorkspaceRegistry

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
