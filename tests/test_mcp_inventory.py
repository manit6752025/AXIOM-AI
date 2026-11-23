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


def test_mcp_inventory_api_maps_missing_server_to_not_found(monkeypatch):
    from fastapi import HTTPException
    from axiom.api import mcp_inventory as api_module

    async def missing(_name: str):
        raise mcp_inventory.McpInventoryError("not found", code="not_found")

    monkeypatch.setattr(api_module, "inspect_mcp_server", missing)
    with pytest.raises(HTTPException) as response:
        asyncio.run(api_module.inspect_mcp_server_handshake("missing"))

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
        "connection_state": "not_inspected",
    }]
    assert "starts the registered process" in result["inspection_note"]
    assert registry.path.exists()


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


def test_handshake_rejects_unknown_server_and_invalid_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _add_server()

    with pytest.raises(mcp_inventory.McpInventoryError) as missing:
        asyncio.run(mcp_inventory.inspect_mcp_server("unknown"))
    assert missing.value.code == "not_found"

    with pytest.raises(mcp_inventory.McpInventoryError) as invalid_timeout:
        asyncio.run(mcp_inventory.inspect_mcp_server("Local MCP", timeout_seconds=30))
    assert invalid_timeout.value.code == "invalid_timeout"
