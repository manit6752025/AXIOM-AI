"""High-signal contract tests for AXIOM's terminal-first product surface.

These tests deliberately exercise only local behavior.  They do not start a
real daemon process, call a vendor service, or install a developer tool.
"""

from __future__ import annotations

import json
import re
from http.client import HTTPConnection
from pathlib import Path

import pytest
from typer.testing import CliRunner

import axiom.cli.main as cli_main
import axiom.daemon as daemon

LOCAL_ACTIONS = daemon.LOCAL_ACTIONS
serve_in_thread = daemon.serve_in_thread
validate_bind_host = daemon.validate_bind_host


def _get_json(host: str, port: int, path: str) -> tuple[int, dict[str, object]]:
    connection = HTTPConnection(host, port, timeout=2)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload
    finally:
        connection.close()


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_daemon_allocates_distinct_free_loopback_ports_and_reports_health():
    first_server, first_thread = serve_in_thread(host="127.0.0.1", port=0)
    second_server, second_thread = serve_in_thread(host="127.0.0.1", port=0)
    try:
        first_host, first_port = first_server.server_address[:2]
        second_host, second_port = second_server.server_address[:2]

        assert first_host == second_host == "127.0.0.1"
        assert first_port > 0
        assert second_port > 0
        assert first_port != second_port

        status, payload = _get_json(second_host, second_port, "/health")

        assert status == 200
        assert payload == {
            "name": "AXIOM",
            "status": "ok",
            "version": payload["version"],
        }
        assert isinstance(payload["version"], str) and payload["version"]
    finally:
        _stop(second_server, second_thread)
        _stop(first_server, first_thread)


def test_daemon_requires_explicit_opt_in_for_non_loopback_binding():
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
    assert validate_bind_host("::1") == "::1"

    with pytest.raises(ValueError, match="local-only"):
        validate_bind_host("0.0.0.0")

    assert validate_bind_host("0.0.0.0", allow_network=True) == "0.0.0.0"


def test_bare_command_lists_every_action_before_starting_the_daemon(
    monkeypatch: pytest.MonkeyPatch,
):
    startup_calls: list[bool] = []
    daemon_calls: list[dict[str, object]] = []

    def fake_startup(*, include_next_steps: bool = False, **_kwargs: object):
        startup_calls.append(include_next_steps)
        return True

    def fake_daemon(**kwargs: object) -> None:
        daemon_calls.append(kwargs)

    monkeypatch.setattr(cli_main, "show_startup", fake_startup)
    monkeypatch.setattr(cli_main, "run_daemon", fake_daemon)

    result = CliRunner().invoke(cli_main.app, [])

    assert result.exit_code == 0, result.stdout
    normalized_output = " ".join(result.stdout.split())
    compact_output = re.sub(r"[^a-z0-9]+", "", normalized_output.lower())
    assert startup_calls == [False]
    assert daemon_calls == [{}]
    assert "AXIOM inspects models" in normalized_output
    assert "Local daemon" in normalized_output

    for action in LOCAL_ACTIONS:
        assert action["command"] in normalized_output
        compact_description = re.sub(
            r"[^a-z0-9]+", "", action["description"].lower()
        )
        assert compact_description in compact_output


@pytest.mark.parametrize("installed", [False, True])
def test_headroom_detection_is_observational_when_catalogued(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed: bool,
):
    """Headroom detection stays observational and never installs or runs it."""

    headroom = next(
        (tool for tool in daemon.TOOL_CATALOG if tool.id == "headroom"),
        None,
    )
    assert headroom is not None, "Headroom must remain discoverable in the catalog"
    if not headroom.executable_names:
        pytest.skip("Headroom is catalogued as API-only, not as a PATH executable")

    executable = headroom.executable_names[0]
    which_calls: list[str] = []

    def fake_which(name: str) -> str | None:
        which_calls.append(name)
        return "/fake/bin/headroom" if installed and name == executable else None

    monkeypatch.setattr(daemon.shutil, "which", fake_which)

    snapshot = daemon._tool_snapshot()
    entry = next(tool for tool in snapshot if tool["id"] == headroom.id)
    assert entry["available"] is installed
    assert entry["executable"] == ("/fake/bin/headroom" if installed else None)
    assert executable in which_calls

    payload = daemon.local_summary(tmp_path)
    assert (headroom.name in payload["tools"]["names"]) is installed
