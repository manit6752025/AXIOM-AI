from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.tool_permissions import ToolPermissionStore

router = APIRouter(prefix="/api/security", tags=["security"])


class ToolProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    allowed_tools: list[str] = Field(max_length=32)


class ActiveProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=48)


@router.post("/tool-profiles/{profile_name}", status_code=201)
def create_tool_profile(profile_name: str) -> dict[str, Any]:
    try:
        profile = ToolPermissionStore().create_profile(profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Tool permission profile could not be created.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("mcp.profile.created", "Created a default-deny MCP profile")
    return {"profile": profile}


@router.get("/tool-permissions")
def get_tool_permissions() -> dict[str, Any]:
    try:
        return ToolPermissionStore().list()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Tool permission policy is invalid.") from exc


@router.put("/tool-profiles/{profile_name}")
def save_tool_profile(profile_name: str, request: ToolProfileInput) -> dict[str, Any]:
    try:
        profile = ToolPermissionStore().save_profile(profile_name, request.allowed_tools)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Tool permission profile could not be saved.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("mcp.permissions.saved", "Updated an MCP tool profile")
    return {"profile": profile}


@router.put("/active-profile")
def set_active_profile(request: ActiveProfileInput) -> dict[str, str]:
    try:
        ToolPermissionStore().set_active(request.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Active MCP profile could not be changed.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("mcp.profile.activated", "Changed the active MCP profile")
    return {"active_profile": request.name}
