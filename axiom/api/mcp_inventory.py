"""Workspace-local MCP server inventory and explicit handshake API."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from axiom.core.mcp_inventory import McpInventoryError, inspect_mcp_server, list_mcp_servers

router = APIRouter(prefix="/api/mcp", tags=["mcp-inventory"])


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
