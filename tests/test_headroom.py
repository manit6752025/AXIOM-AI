import json
from types import SimpleNamespace

from typer.testing import CliRunner

import axiom.daemon as daemon_module
import axiom.cli.headroom as headroom_cli
import axiom.cli.tools as tools_cli
from axiom.cli.main import app

runner = CliRunner()


def test_headroom_status_only_checks_path_and_offers_documented_commands(monkeypatch):
    calls: list[object] = []

    monkeypatch.setattr(headroom_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        headroom_cli.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = runner.invoke(app, ["headroom"])

    assert result.exit_code == 0, result.stdout
    assert "not found on PATH" in result.stdout
    assert "optional local proxy" in result.stdout
    assert "headroom doctor" in result.stdout
    assert "headroom proxy" in result.stdout
    assert "headroom dashboard" in result.stdout
    assert calls == []


def test_headroom_status_json_is_secret_free_and_machine_readable(monkeypatch):
    monkeypatch.setattr(headroom_cli.shutil, "which", lambda _name: "/usr/local/bin/headroom")

    result = runner.invoke(app, ["headroom", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["available"] is True
    assert payload["executable"] == "/usr/local/bin/headroom"
    assert payload["kind"] == "optional-local-proxy"
    assert payload["commands"] == {
        "dashboard": "headroom dashboard",
        "doctor": "headroom doctor",
        "proxy": "headroom proxy --port 8787",
    }
    assert "OPENAI_API_KEY" not in result.stdout
    assert "HEADROOM_API_KEY" not in result.stdout


def test_headroom_commands_are_explicit_documented_passthroughs(monkeypatch):
    calls: list[tuple[list[str], bool]] = []

    monkeypatch.setattr(headroom_cli.shutil, "which", lambda _name: "/tmp/headroom")

    def fake_run(command, *, check):
        calls.append((command, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(headroom_cli.subprocess, "run", fake_run)

    for arguments in (
        ["headroom", "doctor"],
        ["headroom", "proxy", "--port", "9898"],
        ["headroom", "dashboard"],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.stdout

    assert calls == [
        (["/tmp/headroom", "doctor"], False),
        (["/tmp/headroom", "proxy", "--port", "9898"], False),
        (["/tmp/headroom", "dashboard"], False),
    ]


def test_summary_reports_headroom_as_missing_without_running_or_installing_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon_module.shutil, "which", lambda _name: None)

    payload = daemon_module.local_summary(tmp_path)

    assert "Headroom" in payload["tools"]["missing"]


def test_tools_install_does_not_auto_execute_headroom_uv_candidate(monkeypatch):
    called = False

    monkeypatch.setattr(tools_cli.platform, "system", lambda: "Linux")

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Headroom setup must remain explicit")

    monkeypatch.setattr(tools_cli, "execute_install_plan", fail_if_called)

    result = runner.invoke(app, ["tools", "install", "headroom", "--yes"])

    assert result.exit_code == 0, result.stdout
    assert "Skipped" in result.stdout
    assert called is False
