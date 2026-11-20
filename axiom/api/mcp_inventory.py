"""Workspace-local MCP server inventory and explicit handshake API."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from axiom.core.integrations import IntegrationRegistry
from axiom.core.mcp_inventory import McpInventoryError, inspect_mcp_server, list_mcp_servers

router = APIRouter(prefix="/api/mcp", tags=["mcp-inventory"])


class McpServerTrustUpdate(BaseModel):
    trust_status: Literal["unverified", "trusted", "blocked"]
    owner_type: Literal["unknown", "workspace", "external"] | None = None
    owner_label: str | None = Field(default=None, max_length=96)


@router.patch("/servers/{name}/trust")
def update_mcp_server_trust(name: str, request: McpServerTrustUpdate) -> dict[str, Any]:
    """Persist explicit trust and user-declared ownership for an MCP server."""
    try:
        registry = IntegrationRegistry(create=False)
        existing = next(
            (
                item for item in registry.list()
                if item.type.value == "mcp" and item.name.casefold() == name.casefold()
            ),
            None,
        )
        if existing is None:
            raise LookupError(name)
        owner_type = request.owner_type if request.owner_type is not None else existing.owner_type
        owner_label = (
            request.owner_label
            if "owner_label" in request.model_fields_set
            else existing.owner_label
        )
        updated = registry.update_mcp_trust(
            name,
            trust_status=request.trust_status,
            owner_type=owner_type,
            owner_label=owner_label,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="MCP server not found.") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="MCP trust metadata could not be saved safely.") from exc
    return {
        "name": updated.name,
        "trust_status": updated.trust_status,
        "owner_type": updated.owner_type,
        "owner_label": updated.owner_label,
        "ownership_verified": False,
    }



@router.get("/servers")
def get_mcp_servers() -> dict[str, Any]:
    """Return the passive MCP server inventory without spawning processes."""
    try:
        return list_mcp_servers()
    except McpInventoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/servers/{name}/handshake")
async def inspect_mcp_server_handshake(name: str) -> dict[str, Any]:
    """Explicitly launch a registered stdio server and negotiate protocol metadata."""
    try:
        return await inspect_mcp_server(name)
    except McpInventoryError as exc:
        status_code = {
            "not_found": 404,
            "invalid_name": 422,
            "invalid_timeout": 422,
            "missing_command": 409,
            "server_unverified": 409,
            "server_blocked": 403,
        }.get(exc.code, 503)
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/servers/{name}/schemas")
async def inspect_mcp_server_schemas(name: str) -> dict[str, Any]:
    """Explicitly list bounded tool/resource schemas without invoking server capabilities."""
    from axiom.core.mcp_inventory import inspect_mcp_schemas

    try:
        return await inspect_mcp_schemas(name)
    except McpInventoryError as exc:
        status_code = {
            "not_found": 404,
            "invalid_name": 422,
            "invalid_timeout": 422,
            "missing_command": 409,
        }.get(exc.code, 503)
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
