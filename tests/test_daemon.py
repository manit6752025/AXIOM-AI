import json
from io import StringIO
from http.client import HTTPConnection
from pathlib import Path

import pytest

import axiom.daemon as daemon_module

daemon_ready_payload = daemon_module.daemon_ready_payload
run_daemon = daemon_module.run_daemon
serve_in_thread = daemon_module.serve_in_thread
start_daemon = daemon_module.start_daemon
validate_bind_host = daemon_module.validate_bind_host


def get_json(host: str, port: int, path: str) -> tuple[int, dict]:
    connection = HTTPConnection(host, port, timeout=2)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload
    finally:
        connection.close()


@pytest.fixture()
def daemon_server():
    server, thread = serve_in_thread()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_daemon_chooses_an_available_loopback_port(daemon_server):
    host, port = daemon_server.server_address[:2]

    assert host == "127.0.0.1"
    assert port > 0

    status, payload = get_json(host, port, "/health")

    assert status == 200
    assert payload["name"] == "AXIOM"
    assert payload["status"] == "ok"
    assert payload["version"]


def test_daemon_ready_payload_contains_absolute_inspection_urls(daemon_server):
    payload = daemon_ready_payload(daemon_server)

    assert payload["status"] == "ready"
    assert payload["url"].startswith("http://127.0.0.1:")
    assert payload["endpoints"]["health"] == f"{payload['url']}/health"
    assert payload["endpoints"]["summary"] == f"{payload['url']}/api/summary"
    assert payload["next_commands"] == [
        "axiom summary --json",
        f"curl -sS {payload['url']}/health",
        f"curl -sS {payload['url']}/api/summary",
    ]


def test_run_daemon_can_emit_machine_readable_lifecycle_records(monkeypatch):
    server = start_daemon()

    def stop_after_ready():
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "serve_forever", stop_after_ready)
    monkeypatch.setattr(daemon_module, "start_daemon", lambda **_kwargs: server)

    output = StringIO()
    run_daemon(output=output, machine_readable=True)
    records = [json.loads(line) for line in output.getvalue().splitlines()]

    assert records[0]["status"] == "ready"
    assert records[0]["endpoints"]["health"].endswith("/health")
    assert records[-1] == {
        "name": "AXIOM",
        "service": "local-daemon",
        "status": "stopped",
    }


def test_daemon_info_describes_the_local_capabilities(
    daemon_server, tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "axiom.yaml").write_text("name: local\n", encoding="utf-8")
    host, port = daemon_server.server_address[:2]

    status, payload = get_json(host, port, "/api/info")

    assert status == 200
    assert payload["service"] == "local-daemon"
    assert payload["port"] == port
    assert payload["project_config"] is True
    assert "model-inspection" in payload["capabilities"]
    assert payload["endpoints"]["actions"] == "/api/actions"


def test_daemon_exposes_local_actions_hardware_and_tool_presence(daemon_server):
    host, port = daemon_server.server_address[:2]

    status, actions = get_json(host, port, "/api/actions")
    assert status == 200
    assert actions["actions"][0]["command"] == "axiom guide"

    status, hardware = get_json(host, port, "/api/hardware")
    assert status == 200
    assert hardware["hardware"]["cpu_cores"] >= 1
    assert "ram_gb" in hardware["hardware"]

    status, tools = get_json(host, port, "/api/tools")
    assert status == 200
    assert any(tool["id"] == "codex" for tool in tools["tools"])
    assert any(tool["id"] == "headroom" for tool in tools["tools"])


def test_daemon_summary_recommends_a_safe_next_command(
    daemon_server,
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "axiom.yaml").write_text("project:\n  name: local\n", encoding="utf-8")

    host, port = daemon_server.server_address[:2]
    status, payload = get_json(host, port, "/api/summary")

    assert status == 200
    assert payload["project"]["detected"] is True
    assert "README.md" in payload["project"]["missing_paths"]
    assert payload["recommended"]["command"] == "axiom config validate"
    assert payload["tools"]["total"] >= payload["tools"]["available"]


def test_daemon_returns_a_small_not_found_payload(daemon_server):
    host, port = daemon_server.server_address[:2]

    status, payload = get_json(host, port, "/does-not-exist")

    assert status == 404
    assert payload["error"] == "not_found"
    assert "/api/actions" in payload["hint"]


def test_daemon_rejects_non_loopback_hosts_by_default():
    with pytest.raises(ValueError, match="local-only"):
        validate_bind_host("0.0.0.0")

    assert validate_bind_host("0.0.0.0", allow_network=True) == "0.0.0.0"
