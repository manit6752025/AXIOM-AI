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
from itertools import islice
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


def _command_parts(command: str | None, *, windows: bool | None = None) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        return []
    if len(command) > MAX_COMMAND_CHARS or "\x00" in command:
        raise McpInventoryError("Configured MCP command exceeds the supported limit.")
    windows_parser = os.name == "nt" if windows is None else windows
    try:
        parts = shlex.split(command, posix=not windows_parser)
        if windows_parser:
            # shlex preserves outer quotes in non-POSIX mode; remove only a
            # matching pair so spaces and backslashes in Windows paths survive.
            parts = [
                part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'} else part
                for part in parts
            ]
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
            "trust_status": integration.trust_status,
            "owner_type": integration.owner_type,
            "owner_label": _safe_text(integration.owner_label),
            "ownership_verified": False,
            "connection_state": integration.connection_status,
            "last_inspected_at": integration.last_inspected_at,
            "last_connection_error": integration.last_connection_error,
        })

    return {
        "schema_version": 1,
        "total": len(servers),
        "servers": servers,
        "inspection_note": (
            "Inventory is passive. Only servers explicitly marked trusted can be "
            "started for a time-limited handshake or schema inspection."
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
    integration = matches[0]
    if integration.trust_status == "blocked":
        raise McpInventoryError(
            "This MCP server is blocked by the workspace trust policy.",
            code="server_blocked",
        )
    if integration.trust_status != "trusted":
        raise McpInventoryError(
            "This MCP server is unverified. Explicitly mark it trusted before starting the process.",
            code="server_unverified",
        )
    return integration


def _record_connection_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status")
    error_code = result.get("error_code")
    connection_status = (
        "connected" if status == "ok"
        else "timeout" if isinstance(error_code, str) and error_code.endswith("timeout")
        else "failed"
    )
    try:
        updated = _registry().record_mcp_connection(
            name,
            connection_status=connection_status,
            last_connection_error=error_code if status != "ok" else None,
        )
    except Exception:
        return {"connection_state_persisted": False}
    return {
        "connection_state_persisted": True,
        "connection_state": updated.connection_status,
        "last_inspected_at": updated.last_inspected_at,
        "last_connection_error": updated.last_connection_error,
    }


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
        result = {
            "status": "timeout",
            "error_code": "handshake_timeout",
            "message": "The MCP server did not complete protocol negotiation before the timeout.",
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "tool_execution_performed": False,
            "schemas_enumerated": False,
        }
        result.update(_record_connection_result(integration.name, result))
        return result
    except Exception:
        # Do not return child stderr, command arguments, or exception details:
        # they may contain user paths, prompts, or credentials.
        result = {
            "status": "error",
            "error_code": "handshake_failed",
            "message": "The MCP server could not complete protocol negotiation.",
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "tool_execution_performed": False,
            "schemas_enumerated": False,
        }
        result.update(_record_connection_result(integration.name, result))
        return result

    result.update(_record_connection_result(integration.name, result))
    return {
        "name": _safe_text(integration.name) or "Unnamed MCP server",
        "transport": "stdio",
        **result,
    }


MAX_SCHEMA_PAGES = 4
MAX_SCHEMA_ITEMS = 100
MAX_SCHEMA_DEPTH = 8
MAX_SCHEMA_NODES = 512
MAX_SCHEMA_TEXT_CHARS = 512
_SAFE_SCHEMA_KEYS = frozenset({
    "type", "title", "description", "format", "pattern",
    "properties", "required", "items", "additionalProperties",
    "enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems", "uniqueItems",
    "oneOf", "anyOf", "allOf", "not", "$defs", "definitions",
})


def _safe_schema(value: object, *, depth: int = 0, budget: list[int] | None = None) -> object:
    """Keep a bounded JSON-Schema subset and deliberately omit defaults/examples."""
    if budget is None:
        budget = [MAX_SCHEMA_NODES]
    if budget[0] <= 0 or depth > MAX_SCHEMA_DEPTH:
        return None
    budget[0] -= 1

    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in islice(value.items(), 64):
            if not isinstance(key, str) or key not in _SAFE_SCHEMA_KEYS:
                continue
            if key in {"title", "description", "format", "pattern", "type"}:
                if isinstance(item, str):
                    result[key] = item[:MAX_SCHEMA_TEXT_CHARS]
                continue
            if key == "required":
                if isinstance(item, list):
                    result[key] = [
                        name[:96] for name in item[:64]
                        if isinstance(name, str) and name.isprintable()
                    ]
                continue
            if key == "enum":
                if isinstance(item, list):
                    result[key] = [
                        entry[:128] if isinstance(entry, str) else entry
                        for entry in item[:32]
                        if isinstance(entry, (str, int, float, bool)) or entry is None
                    ]
                continue
            if key in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                       "minLength", "maxLength", "minItems", "maxItems", "uniqueItems"}:
                if isinstance(item, (int, float, bool)):
                    result[key] = item
                continue
            if key == "properties" or key in {"$defs", "definitions"}:
                if isinstance(item, dict):
                    result[key] = {
                        name[:96]: _safe_schema(schema, depth=depth + 1, budget=budget)
                        for name, schema in islice(item.items(), 64)
                        if isinstance(name, str) and name.isprintable()
                    }
                continue
            if key == "additionalProperties" and isinstance(item, bool):
                result[key] = item
                continue
            if key in {"items", "additionalProperties", "oneOf", "anyOf", "allOf", "not"}:
                if isinstance(item, list):
                    result[key] = [
                        _safe_schema(entry, depth=depth + 1, budget=budget)
                        for entry in item[:32]
                    ]
                elif isinstance(item, dict):
                    result[key] = _safe_schema(item, depth=depth + 1, budget=budget)
        return result

    if isinstance(value, list):
        return [
            _safe_schema(item, depth=depth + 1, budget=budget)
            for item in value[:32]
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:MAX_SCHEMA_TEXT_CHARS] if isinstance(value, str) else value
    return None


def _safe_resource_uri(value: object) -> str | None:
    """Show resource identity without credentials, query values, or local directories."""
    from urllib.parse import urlsplit

    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[REDACTED URI]"
    scheme = parsed.scheme[:32]
    if not scheme:
        return "[REDACTED URI]"
    if scheme == "file":
        filename = Path(parsed.path).name
        return f"file://…/{filename[:128]}" if filename else "file://…"
    if parsed.hostname:
        host = parsed.hostname[:128]
        try:
            parsed_port = parsed.port
        except ValueError:
            return "[REDACTED URI]"
        port = f":{parsed_port}" if parsed_port is not None else ""
        return f"{scheme}://{host}{port}/…"
    return f"{scheme}:…"


async def _collect_schema_pages(client: Any, method_name: str, collection_name: str) -> tuple[list[Any], bool]:
    collected: list[Any] = []
    cursor: str | None = None
    truncated = False
    for page_number in range(MAX_SCHEMA_PAGES):
        method = getattr(client, method_name)
        page = await method(cursor=cursor)
        entries = getattr(page, collection_name, None)
        if entries is None and collection_name == "resourceTemplates":
            entries = getattr(page, "resource_templates", None)
        entries = entries or []
        for entry in entries:
            if len(collected) >= MAX_SCHEMA_ITEMS:
                truncated = True
                break
            collected.append(entry)
        cursor = getattr(page, "next_cursor", None)
        if truncated:
            break
        if not cursor:
            break
        if page_number == MAX_SCHEMA_PAGES - 1:
            truncated = True
    return collected, truncated


def _tool_schema_record(tool: Any) -> dict[str, Any]:
    input_schema = _safe_schema(getattr(tool, "input_schema", None))
    output_schema = _safe_schema(getattr(tool, "output_schema", None))
    annotations = getattr(tool, "annotations", None)
    return {
        "name": _safe_text(getattr(tool, "name", None)) or "Unnamed tool",
        "title": _safe_text(getattr(tool, "title", None)),
        "description": _safe_text(getattr(tool, "description", None)),
        "input_schema": input_schema if isinstance(input_schema, dict) else {},
        "output_schema": output_schema if isinstance(output_schema, dict) else None,
        "annotations": {
            key: getattr(annotations, key, None)
            for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
            if isinstance(getattr(annotations, key, None), bool)
        },
    }


def _resource_schema_record(resource: Any) -> dict[str, Any]:
    return {
        "name": _safe_text(getattr(resource, "name", None)),
        "uri": _safe_resource_uri(getattr(resource, "uri", None)),
        "description": _safe_text(getattr(resource, "description", None)),
        "mime_type": _safe_text(getattr(resource, "mimeType", getattr(resource, "mime_type", None))),
        "size_bytes": (
            getattr(resource, "size", None)
            if isinstance(getattr(resource, "size", None), int)
            and not isinstance(getattr(resource, "size", None), bool)
            and 0 <= getattr(resource, "size", None) <= 2**53 - 1
            else None
        ),
    }


def _resource_template_record(template: Any) -> dict[str, Any]:
    return {
        "name": _safe_text(getattr(template, "name", None)),
        "uri_template": _safe_resource_uri(getattr(template, "uriTemplate", getattr(template, "uri_template", None))),
        "description": _safe_text(getattr(template, "description", None)),
        "mime_type": _safe_text(getattr(template, "mimeType", getattr(template, "mime_type", None))),
    }


async def _perform_schema_browse(parts: list[str]) -> dict[str, Any]:
    parameters = StdioServerParameters(command=parts[0], args=parts[1:])
    with open(os.devnull, "w", encoding="utf-8") as stderr_sink:
        transport = stdio_client(parameters, errlog=stderr_sink)
        async with Client(transport, mode="auto", read_timeout_seconds=10.0) as client:
            capabilities = getattr(client, "server_capabilities", None)
            supports_tools = getattr(capabilities, "tools", None) is not None
            supports_resources = getattr(capabilities, "resources", None) is not None

            tool_records: list[dict[str, Any]] = []
            resource_records: list[dict[str, Any]] = []
            template_records: list[dict[str, Any]] = []
            resource_templates_supported = False
            truncation = {"tools": False, "resources": False, "resource_templates": False}

            if supports_tools:
                tools, truncation["tools"] = await _collect_schema_pages(
                    client, "list_tools", "tools"
                )
                tool_records = [_tool_schema_record(tool) for tool in tools]
            if supports_resources:
                resources, truncation["resources"] = await _collect_schema_pages(
                    client, "list_resources", "resources"
                )
                resource_records = [_resource_schema_record(resource) for resource in resources]
                list_templates = getattr(client, "list_resource_templates", None)
                if callable(list_templates):
                    try:
                        templates, truncation["resource_templates"] = await _collect_schema_pages(
                            client, "list_resource_templates", "resourceTemplates"
                        )
                    except Exception:
                        # Resource templates are an optional MCP listing. A server
                        # may expose resources while omitting this method.
                        templates = []
                    else:
                        resource_templates_supported = True
                    template_records = [_resource_template_record(item) for item in templates]

            server_info = getattr(client, "server_info", None)
            return {
                "status": "ok",
                "protocol_version": _safe_text(getattr(client, "protocol_version", None)),
                "server_name": _safe_text(getattr(server_info, "name", None)),
                "server_version": _safe_text(getattr(server_info, "version", None)),
                "tools_supported": supports_tools,
                "resources_supported": supports_resources,
                "resource_templates_supported": resource_templates_supported,
                "tools": tool_records,
                "resources": resource_records,
                "resource_templates": template_records,
                "truncated": truncation,
                "tool_execution_performed": False,
                "resource_read_performed": False,
                "prompt_execution_performed": False,
                "schema_browse_scope": "list metadata only; no tools/call or resources/read requests",
            }


async def inspect_mcp_schemas(
    name: str,
    *,
    timeout_seconds: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Explicitly inspect bounded MCP tool/resource schemas without invoking them."""
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
            f"Schema inspection timeout must be between 1 and {MAX_HANDSHAKE_TIMEOUT_SECONDS:g} seconds.",
            code="invalid_timeout",
        )
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            _perform_schema_browse(parts),
            timeout=float(timeout_seconds),
        )
    except TimeoutError:
        result = {
            "status": "timeout",
            "error_code": "schema_inspection_timeout",
            "message": "MCP schema inspection did not complete before the timeout.",
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "tool_execution_performed": False,
            "resource_read_performed": False,
            "prompt_execution_performed": False,
        }
        result.update(_record_connection_result(integration.name, result))
        return result
    except Exception:
        result = {
            "status": "error",
            "error_code": "schema_inspection_failed",
            "message": "MCP schema metadata could not be inspected safely.",
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "tool_execution_performed": False,
            "resource_read_performed": False,
            "prompt_execution_performed": False,
        }
        result.update(_record_connection_result(integration.name, result))
        return result
    result.update(_record_connection_result(integration.name, result))
    return {
        "name": _safe_text(integration.name) or "Unnamed MCP server",
        "transport": "stdio",
        "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
        **result,
    }
