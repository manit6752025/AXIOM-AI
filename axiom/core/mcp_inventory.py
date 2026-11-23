"""Read-only MCP server inventory and time-bounded protocol inspection.

Only explicitly registered stdio servers are inspected. A handshake launches the
configured process without a shell, uses the MCP SDK's minimal child environment,
and never lists schemas or invokes tools.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from axiom.core.integrations import IntegrationRegistry, IntegrationType

MAX_REGISTRY_BYTES = 512 * 1024
MAX_COMMAND_CHARS = 8192
MAX_COMMAND_PARTS = 64
DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 10.0
MAX_HANDSHAKE_TIMEOUT_SECONDS = 15.0
_METADATA_CHARS = 128


class McpInventoryError(ValueError):
    """An MCP inventory entry is invalid, missing, or unsafe to inspect."""

    def __init__(self, message: str, *, code: str = "invalid_inventory") -> None:
        super().__init__(message)
        self.code = code


def _registry() -> IntegrationRegistry:
    registry = IntegrationRegistry(create=False)
    try:
        if registry.root.is_symlink() or registry.path.is_symlink():
            raise McpInventoryError("MCP integration registry path is unsafe.")
        if registry.root.exists() and not registry.root.is_dir():
            raise McpInventoryError("MCP integration registry directory is invalid.")
        if registry.path.exists():
            if not registry.path.is_file() or registry.path.stat().st_size > MAX_REGISTRY_BYTES:
                raise McpInventoryError("MCP integration registry is not a bounded regular file.")
    except OSError as exc:
        raise McpInventoryError("MCP integration registry could not be inspected safely.") from exc
    return registry


def _command_parts(command: str | None) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        return []
    if len(command) > MAX_COMMAND_CHARS or "\x00" in command:
        raise McpInventoryError("Configured MCP command exceeds the supported limit.")
    try:
        parts = shlex.split(command, posix=True)
    except ValueError as exc:
        raise McpInventoryError("Configured MCP command quoting is invalid.") from exc
    if not parts or len(parts) > MAX_COMMAND_PARTS:
        raise McpInventoryError("Configured MCP command has an invalid number of arguments.")
    return parts


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = "".join(character for character in value if character.isprintable()).strip()
    if not cleaned:
        return None
    return cleaned[:_METADATA_CHARS]


def list_mcp_servers() -> dict[str, Any]:
    """Return configured MCP entries without starting or connecting to processes."""
    registry = _registry()
    try:
        integrations = registry.list()
    except (OSError, TypeError, ValueError) as exc:
        raise McpInventoryError("MCP integration registry could not be read safely.") from exc

    servers: list[dict[str, Any]] = []
    for integration in integrations:
        if integration.type is not IntegrationType.MCP:
            continue
        try:
            parts = _command_parts(integration.command)
        except McpInventoryError:
            parts = []
        servers.append({
            "name": _safe_text(integration.name) or "Unnamed MCP server",
            "transport": "stdio",
            "configured": bool(parts),
            "executable": Path(parts[0]).name[:_METADATA_CHARS] if parts else None,
            "argument_count": max(0, len(parts) - 1),
            "connection_state": "not_inspected",
        })

    return {
        "schema_version": 1,
        "total": len(servers),
        "servers": servers,
        "inspection_note": (
            "Inventory is passive. A handshake starts the registered process only "
            "after an explicit inspection request."
        ),
    }


def _find_server(name: str):
    if not isinstance(name, str) or not name.strip() or len(name) > _METADATA_CHARS:
        raise McpInventoryError("MCP server name is invalid.", code="invalid_name")
    registry = _registry()
    try:
        matches = [
            item for item in registry.list()
            if item.type is IntegrationType.MCP and item.name.casefold() == name.strip().casefold()
        ]
    except (OSError, TypeError, ValueError) as exc:
        raise McpInventoryError("MCP integration registry could not be read safely.") from exc
    if not matches:
        raise McpInventoryError("The requested MCP server is not registered.", code="not_found")
    return matches[0]


async def _perform_handshake(parts: list[str], timeout_seconds: float) -> dict[str, Any]:
    parameters = StdioServerParameters(
        command=parts[0],
        args=parts[1:],
        # Intentionally do not pass the AXIOM process environment or credentials.
    )
    started = time.monotonic()
    with open(os.devnull, "w", encoding="utf-8") as stderr_sink:
        transport = stdio_client(parameters, errlog=stderr_sink)
        async with Client(
            transport,
            mode="auto",
            read_timeout_seconds=min(timeout_seconds, MAX_HANDSHAKE_TIMEOUT_SECONDS),
        ) as client:
            # Entering Client completes the MCP protocol negotiation. Deliberately
            # do not call list_tools/list_resources/list_prompts or call_tool.
            server_info = getattr(client, "server_info", None)
            capabilities = getattr(client, "server_capabilities", None)
            return {
                "status": "ok",
                "protocol_version": _safe_text(getattr(client, "protocol_version", None)),
                "server_name": _safe_text(getattr(server_info, "name", None)),
                "server_version": _safe_text(getattr(server_info, "version", None)),
                "capabilities": {
                    "tools": getattr(capabilities, "tools", None) is not None,
                    "resources": getattr(capabilities, "resources", None) is not None,
                    "prompts": getattr(capabilities, "prompts", None) is not None,
                    "logging": getattr(capabilities, "logging", None) is not None,
                    "completions": getattr(capabilities, "completions", None) is not None,
                },
                "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
                "tool_execution_performed": False,
                "schemas_enumerated": False,
            }


async def inspect_mcp_server(
    name: str,
    *,
    timeout_seconds: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Explicitly launch one registered stdio server and negotiate protocol metadata."""
    integration = _find_server(name)
    parts = _command_parts(integration.command)
    if not parts:
        raise McpInventoryError(
            "The registered MCP server has no valid stdio command.",
            code="missing_command",
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1 <= float(timeout_seconds) <= MAX_HANDSHAKE_TIMEOUT_SECONDS
    ):
        raise McpInventoryError(
            f"Handshake timeout must be between 1 and {MAX_HANDSHAKE_TIMEOUT_SECONDS:g} seconds.",
            code="invalid_timeout",
        )

    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            _perform_handshake(parts, float(timeout_seconds)),
            timeout=float(timeout_seconds),
        )
    except TimeoutError:
        return {
            "status": "timeout",
            "error_code": "handshake_timeout",
            "message": "The MCP server did not complete protocol negotiation before the timeout.",
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "tool_execution_performed": False,
            "schemas_enumerated": False,
        }
    except Exception:
        # Do not return child stderr, command arguments, or exception details:
        # they may contain user paths, prompts, or credentials.
        return {
            "status": "error",
            "error_code": "handshake_failed",
            "message": "The MCP server could not complete protocol negotiation.",
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "tool_execution_performed": False,
            "schemas_enumerated": False,
        }

    return {
        "name": _safe_text(integration.name) or "Unnamed MCP server",
        "transport": "stdio",
        **result,
    }
