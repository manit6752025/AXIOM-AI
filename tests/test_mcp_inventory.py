from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from axiom.core.integrations import AgentIntegration, IntegrationRegistry, IntegrationType
from axiom.core import mcp_inventory


def _add_server(command: str = 'python -u -m example.server --config "private path.json"') -> None:
    IntegrationRegistry().add(
        AgentIntegration(
            name="Local MCP",
            type=IntegrationType.MCP,
            command=command,
            description="Test MCP process",
            trust_status="trusted",
        )
    )


def test_mcp_inventory_api_routes_are_registered_with_expected_methods():
    from axiom.api.mcp_inventory import router

    routes = {
        route.path: route.methods
        for route in router.routes
    }

    assert routes["/api/mcp/servers"] == {"GET"}
    assert routes["/api/mcp/servers/{name}/handshake"] == {"POST"}
    assert routes["/api/mcp/servers/{name}/schemas"] == {"POST"}
    assert routes["/api/mcp/servers/{name}/trust"] == {"PATCH"}


def test_mcp_inventory_api_maps_missing_server_to_not_found(monkeypatch):
    from fastapi import HTTPException
    from axiom.api import mcp_inventory as api_module

    async def missing(_name: str):
        raise mcp_inventory.McpInventoryError("not found", code="not_found")

    monkeypatch.setattr(api_module, "inspect_mcp_server", missing)
    with pytest.raises(HTTPException) as response:
        asyncio.run(api_module.inspect_mcp_server_handshake("missing"))

    assert response.value.status_code == 404


def test_mcp_schema_api_maps_unknown_server_to_not_found(monkeypatch):
    from fastapi import HTTPException
    from axiom.api import mcp_inventory as api_module

    async def missing(_name: str):
        raise mcp_inventory.McpInventoryError("not found", code="not_found")

    monkeypatch.setattr(mcp_inventory, "inspect_mcp_schemas", missing)
    with pytest.raises(HTTPException) as response:
        asyncio.run(api_module.inspect_mcp_server_schemas("missing"))

    assert response.value.status_code == 404


def test_inventory_is_passive_bounded_and_only_lists_mcp_servers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = IntegrationRegistry()
    registry.add(AgentIntegration("Local MCP", IntegrationType.MCP, command="python -m example"))
    registry.add(AgentIntegration("Local CLI", IntegrationType.CLI, command="tool"))
    registry.add(AgentIntegration("Remote API", IntegrationType.HTTP, url="https://example.invalid"))

    result = mcp_inventory.list_mcp_servers()

    assert result["total"] == 1
    assert result["servers"] == [{
        "name": "Local MCP",
        "transport": "stdio",
        "configured": True,
        "executable": "python",
        "argument_count": 2,
        "trust_status": "unverified",
        "owner_type": "unknown",
        "owner_label": None,
        "ownership_verified": False,
        "connection_state": "not_inspected",
        "last_inspected_at": None,
        "last_connection_error": None,
    }]
    assert "starts the registered process" in result["inspection_note"]
    assert registry.path.exists()


def test_windows_command_parser_preserves_quoted_paths_and_backslashes():
    command = '"C:\\Program Files\\AXIOM\\python.exe" -m sample.server --config "D:\\model data\\config.json"'

    parts = mcp_inventory._command_parts(command, windows=True)

    assert parts == [
        "C:\\Program Files\\AXIOM\\python.exe",
        "-m",
        "sample.server",
        "--config",
        "D:\\model data\\config.json",
    ]


def test_inventory_does_not_create_metadata_when_registry_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = mcp_inventory.list_mcp_servers()

    assert result["total"] == 0
    assert not (tmp_path / ".axiom").exists()


def test_inventory_redacts_command_arguments_and_marks_invalid_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    IntegrationRegistry().add(
        AgentIntegration("Bad quoting", IntegrationType.MCP, command='python "unterminated')
    )

    result = mcp_inventory.list_mcp_servers()

    assert result["total"] == 1
    assert result["servers"][0]["configured"] is False
    assert result["servers"][0]["executable"] is None
    assert "unterminated" not in str(result)


def test_handshake_uses_stdio_without_shell_or_inherited_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()
    observed = {}

    class FakeClient:
        def __init__(self, transport, *, mode, read_timeout_seconds):
            observed["transport"] = transport
            observed["mode"] = mode
            observed["read_timeout_seconds"] = read_timeout_seconds
            self.protocol_version = "2025-11-25"
            self.server_info = SimpleNamespace(name="Example MCP", version="2.4.1")
            self.server_capabilities = SimpleNamespace(
                tools=object(), resources=None, prompts=object(),
                logging=None, completions=None,
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    def fake_stdio_client(parameters, *, errlog):
        observed["command"] = parameters.command
        observed["args"] = parameters.args
        observed["env"] = parameters.env
        observed["errlog"] = errlog
        return object()

    monkeypatch.setattr(mcp_inventory, "Client", FakeClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", fake_stdio_client)

    result = asyncio.run(mcp_inventory.inspect_mcp_server("local mcp"))

    assert result["status"] == "ok"
    assert result["transport"] == "stdio"
    assert result["name"] == "Local MCP"
    assert result["protocol_version"] == "2025-11-25"
    assert result["server_name"] == "Example MCP"
    assert result["server_version"] == "2.4.1"
    assert result["capabilities"] == {
        "tools": True, "resources": False, "prompts": True,
        "logging": False, "completions": False,
    }
    assert result["tool_execution_performed"] is False
    assert result["schemas_enumerated"] is False
    assert observed["command"] == "python"
    assert observed["args"] == ["-u", "-m", "example.server", "--config", "private path.json"]
    assert observed["env"] is None
    assert observed["mode"] == "auto"
    assert observed["read_timeout_seconds"] == mcp_inventory.DEFAULT_HANDSHAKE_TIMEOUT_SECONDS
    assert observed["errlog"] is not None


def test_handshake_timeout_is_safe_and_never_returns_process_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    class SlowClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            await asyncio.sleep(2)

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(mcp_inventory, "Client", SlowClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", lambda *_args, **_kwargs: object())

    result = asyncio.run(mcp_inventory.inspect_mcp_server("Local MCP", timeout_seconds=1))

    assert result["status"] == "timeout"
    assert result["error_code"] == "handshake_timeout"
    assert result["tool_execution_performed"] is False
    assert result["schemas_enumerated"] is False
    assert "private path.json" not in str(result)


def test_handshake_failures_do_not_disclose_command_or_exception_details(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    class BrokenClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            raise RuntimeError("token=secret-value private path.json")

    monkeypatch.setattr(mcp_inventory, "Client", BrokenClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", lambda *_args, **_kwargs: object())

    result = asyncio.run(mcp_inventory.inspect_mcp_server("Local MCP"))

    assert result["status"] == "error"
    assert result["error_code"] == "handshake_failed"
    assert "secret-value" not in str(result)
    assert "private path.json" not in str(result)


def test_inventory_rejects_nul_and_oversized_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = IntegrationRegistry()
    registry.add(AgentIntegration("NUL command", IntegrationType.MCP, command="python\x00secret"))
    registry.add(AgentIntegration("Large command", IntegrationType.MCP, command="x" * (mcp_inventory.MAX_COMMAND_CHARS + 1)))

    result = mcp_inventory.list_mcp_servers()

    assert result["total"] == 2
    assert all(server["configured"] is False for server in result["servers"])
    assert "secret" not in str(result)


def test_handshake_negotiates_with_real_mcp_stdio_server(tmp_path, monkeypatch):
    import shlex
    import sys

    monkeypatch.chdir(tmp_path)
    server_script = tmp_path / "mcp_probe.py"
    server_script.write_text(
        "from mcp.server import MCPServer\n"
        "server = MCPServer('AXIOM Inventory Probe')\n"
        "if __name__ == '__main__':\n"
        "    server.run()\n",
        encoding="utf-8",
    )
    IntegrationRegistry().add(
        AgentIntegration(
            "Process Probe",
            IntegrationType.MCP,
            command=shlex.join([sys.executable, str(server_script)]),
            trust_status="trusted",
        )
    )

    result = asyncio.run(
        mcp_inventory.inspect_mcp_server("Process Probe", timeout_seconds=10)
    )

    assert result["status"] == "ok", result
    assert result["transport"] == "stdio"
    assert result["server_name"] == "AXIOM Inventory Probe"
    assert result["protocol_version"]
    assert result["tool_execution_performed"] is False
    assert result["schemas_enumerated"] is False




def test_schema_browse_lists_bounded_tool_and_resource_metadata_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()
    calls = {"tools": [], "resources": [], "templates": [], "execute": 0, "read": 0}

    class FakeClient:
        protocol_version = "2025-11-25"
        server_info = SimpleNamespace(name="Schema Probe", version="3.2")
        server_capabilities = SimpleNamespace(tools=object(), resources=object())

        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self, *, cursor=None):
            calls["tools"].append(cursor)
            if cursor is None:
                tools = [SimpleNamespace(
                    name="lookup",
                    title="Lookup",
                    description="Search local metadata",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query", "default": "secret-default"},
                            "token": {"type": "string", "examples": ["secret-example"]},
                        },
                        "required": ["query"],
                        "default": {"token": "secret-value"},
                    },
                    output_schema={"type": "object", "properties": {"count": {"type": "integer"}}},
                    annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
                )]
                return SimpleNamespace(tools=tools, next_cursor="page-2")
            return SimpleNamespace(tools=[SimpleNamespace(
                name="status", title=None, description=None,
                input_schema={"type": "object"}, output_schema=None, annotations=None,
            )], next_cursor=None)

        async def list_resources(self, *, cursor=None):
            calls["resources"].append(cursor)
            return SimpleNamespace(resources=[SimpleNamespace(
                name="Private file",
                uri="file:///home/alice/private/secrets.json",
                description="Metadata only",
                mimeType="application/json",
                size=128,
            )], next_cursor=None)

        async def list_resource_templates(self, *, cursor=None):
            calls["templates"].append(cursor)
            return SimpleNamespace(resource_templates=[SimpleNamespace(
                name="Account",
                uri_template="https://example.test/accounts/{id}?token=private",
                description="Account resource",
                mimeType="application/json",
            )], next_cursor=None)

        async def call_tool(self, *_args, **_kwargs):
            calls["execute"] += 1
            raise AssertionError("Schema browsing must not invoke tools")

        async def read_resource(self, *_args, **_kwargs):
            calls["read"] += 1
            raise AssertionError("Schema browsing must not read resources")

    monkeypatch.setattr(mcp_inventory, "Client", FakeClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", lambda *_args, **_kwargs: object())

    result = asyncio.run(mcp_inventory.inspect_mcp_schemas("Local MCP"))

    assert result["status"] == "ok"
    assert result["protocol_version"] == "2025-11-25"
    assert [item["name"] for item in result["tools"]] == ["lookup", "status"]
    assert result["tools"][0]["annotations"]["readOnlyHint"] is True
    assert result["tools"][0]["input_schema"]["properties"]["query"]["type"] == "string"
    assert "default" not in result["tools"][0]["input_schema"]
    assert "default" not in result["tools"][0]["input_schema"]["properties"]["query"]
    assert "examples" not in result["tools"][0]["input_schema"]["properties"]["token"]
    assert result["resources"][0]["uri"] == "file://…/secrets.json"
    assert result["resource_templates"][0]["uri_template"] == "https://example.test/…"
    assert result["tool_execution_performed"] is False
    assert result["resource_read_performed"] is False
    assert result["prompt_execution_performed"] is False
    assert calls == {"tools": [None, "page-2"], "resources": [None], "templates": [None], "execute": 0, "read": 0}
    assert "secret-value" not in str(result)
    assert "secret-default" not in str(result)
    assert "secret-example" not in str(result)
    assert "/home/alice/private" not in str(result)
    assert "token=private" not in str(result)


def test_schema_browse_respects_capability_flags_and_marks_pagination_truncation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    class FakeClient:
        protocol_version = "2025-11-25"
        server_info = None
        server_capabilities = SimpleNamespace(tools=object(), resources=None)

        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self, *, cursor=None):
            return SimpleNamespace(
                tools=[SimpleNamespace(name=f"tool-{cursor or 'first'}", input_schema={"type": "object"})],
                next_cursor=f"next-{cursor or 'first'}",
            )

        async def list_resources(self, **_kwargs):
            raise AssertionError("Resources are not advertised")

    monkeypatch.setattr(mcp_inventory, "Client", FakeClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mcp_inventory, "MAX_SCHEMA_PAGES", 2)

    result = asyncio.run(mcp_inventory.inspect_mcp_schemas("Local MCP"))

    assert result["tools_supported"] is True
    assert result["resources_supported"] is False
    assert len(result["tools"]) == 2
    assert result["resources"] == []
    assert result["truncated"]["tools"] is True
    assert result["truncated"]["resources"] is False


def test_schema_browse_tolerates_optional_resource_templates_method_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    class FakeClient:
        protocol_version = "2025-11-25"
        server_info = None
        server_capabilities = SimpleNamespace(tools=object(), resources=object())

        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self, *, cursor=None):
            return SimpleNamespace(tools=[SimpleNamespace(
                name="lookup", input_schema={"type": "object"},
            )], next_cursor=None)

        async def list_resources(self, *, cursor=None):
            return SimpleNamespace(resources=[SimpleNamespace(
                name="Catalog", uri="catalog://items",
            )], next_cursor=None)

        async def list_resource_templates(self, *, cursor=None):
            raise RuntimeError("Method not found")

    monkeypatch.setattr(mcp_inventory, "Client", FakeClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", lambda *_args, **_kwargs: object())

    result = asyncio.run(mcp_inventory.inspect_mcp_schemas("Local MCP"))

    assert result["status"] == "ok"
    assert [tool["name"] for tool in result["tools"]] == ["lookup"]
    assert [resource["name"] for resource in result["resources"]] == ["Catalog"]
    assert result["resource_templates"] == []
    assert result["resource_templates_supported"] is False


def test_schema_inspection_timeout_returns_no_child_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    class SlowClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            await asyncio.sleep(2)

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(mcp_inventory, "Client", SlowClient)
    monkeypatch.setattr(mcp_inventory, "stdio_client", lambda *_args, **_kwargs: object())

    result = asyncio.run(mcp_inventory.inspect_mcp_schemas("Local MCP", timeout_seconds=1))

    assert result["status"] == "timeout"
    assert result["tool_execution_performed"] is False
    assert result["resource_read_performed"] is False
    assert "private path.json" not in str(result)



def test_schema_browse_negotiates_real_stdio_tool_resource_and_template_metadata(tmp_path, monkeypatch):
    import shlex
    import sys

    monkeypatch.chdir(tmp_path)
    server_script = tmp_path / "schema_probe.py"
    marker = tmp_path / "capability-executed.txt"
    server_script.write_text(
        "from pathlib import Path\n"
        "from mcp.server import MCPServer\n"
        "server = MCPServer('AXIOM Schema Probe')\n"
        f"marker = Path({str(marker)!r})\n"
        "@server.tool()\n"
        "def search_catalog(query: str) -> dict:\n"
        "    'Search the catalog by name.'\n"
        "    marker.write_text('tool executed', encoding='utf-8')\n"
        "    return {'matches': []}\n"
        "@server.resource('catalog://items')\n"
        "def catalog_items() -> str:\n"
        "    'The catalog item index.'\n"
        "    marker.write_text('resource read', encoding='utf-8')\n"
        "    return 'item index'\n"
        "@server.resource('catalog://items/{item_id}')\n"
        "def catalog_item(item_id: str) -> str:\n"
        "    'A catalog item.'\n"
        "    marker.write_text('resource template read', encoding='utf-8')\n"
        "    return item_id\n"
        "if __name__ == '__main__':\n"
        "    server.run()\n",
        encoding="utf-8",
    )
    IntegrationRegistry().add(
        AgentIntegration(
            "Schema Probe",
            IntegrationType.MCP,
            command=shlex.join([sys.executable, str(server_script)]),
            trust_status="trusted",
        )
    )

    result = asyncio.run(
        mcp_inventory.inspect_mcp_schemas("Schema Probe", timeout_seconds=12)
    )

    assert result["status"] == "ok", result
    assert result["server_name"] == "AXIOM Schema Probe"
    assert [tool["name"] for tool in result["tools"]] == ["search_catalog"]
    assert result["tools"][0]["input_schema"]["properties"]["query"]["type"] == "string"
    assert [resource["name"] for resource in result["resources"]] == ["catalog_items"]
    assert [template["name"] for template in result["resource_templates"]] == ["catalog_item"]
    assert result["tool_execution_performed"] is False
    assert result["resource_read_performed"] is False
    assert result["prompt_execution_performed"] is False
    assert not marker.exists(), "Schema browsing must not execute tool or resource handlers"


def test_handshake_rejects_unknown_server_and_invalid_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    with pytest.raises(mcp_inventory.McpInventoryError) as missing:
        asyncio.run(mcp_inventory.inspect_mcp_server("unknown"))
    assert missing.value.code == "not_found"

    with pytest.raises(mcp_inventory.McpInventoryError) as invalid_timeout:
        asyncio.run(mcp_inventory.inspect_mcp_server("Local MCP", timeout_seconds=30))
    assert invalid_timeout.value.code == "invalid_timeout"
