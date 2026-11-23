from typer.testing import CliRunner

import axiom.cli.tools as cli_tools
from axiom.cli.main import app

runner = CliRunner()


def test_tools_list_is_local_and_machine_readable(monkeypatch):
    monkeypatch.setattr(cli_tools.platform, "system", lambda: "Linux")
    result = runner.invoke(app, ["tools", "list", "--json"])

    assert result.exit_code == 0, result.stdout
    assert '"id": "codex"' in result.stdout
    assert '"id": "zai-glm"' in result.stdout
    assert "credential" not in result.stdout.lower()


def test_tools_plan_shows_vendor_command_without_running_it(monkeypatch):
    monkeypatch.setattr(cli_tools.platform, "system", lambda: "Linux")
    result = runner.invoke(app, ["tools", "plan", "claude-code"])

    assert result.exit_code == 0, result.stdout
    assert "npm install -g @anthropic-ai/claude-code" in result.stdout
    assert "https://" in result.stdout


def test_tools_install_defaults_to_a_preview(monkeypatch):
    monkeypatch.setattr(cli_tools.platform, "system", lambda: "Linux")
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("preview must not execute an installer")

    monkeypatch.setattr(cli_tools, "execute_install_plan", fail_if_called)
    result = runner.invoke(app, ["tools", "install", "claude-code"])

    assert result.exit_code == 0, result.stdout
    assert "AXIOM command" in result.stdout
    assert called is False



def test_tools_doctor_json_reports_local_presence_without_leaking_credentials(monkeypatch):
    monkeypatch.setattr(cli_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_tools.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setenv("OPENAI_API_KEY", "never-return-this-secret-value")
    result = runner.invoke(app, ["tools", "doctor", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = __import__("json").loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["platform"] == "linux"
    codex = next(item for item in payload["tools"] if item["id"] == "codex")
    assert codex["installed_executables"] == ["codex"]
    assert codex["credential_reference_available"] is True
    assert codex["vendor_authentication_verified"] is False
    assert "never-return-this-secret-value" not in result.stdout
    assert "OPENAI_API_KEY" not in result.stdout


def test_tools_plan_json_lists_catalog_candidates_without_executing_them(monkeypatch):
    monkeypatch.setattr(cli_tools.platform, "system", lambda: "Linux")
    def fail(*_args, **_kwargs):
        raise AssertionError("a plan must not execute commands")
    monkeypatch.setattr(cli_tools.subprocess, "run", fail)
    result = runner.invoke(app, ["tools", "plan", "claude-code", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = __import__("json").loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["tool"]["id"] == "claude-code"
    assert payload["platform"] == "linux"
    assert any(item["command"] == "npm install -g @anthropic-ai/claude-code" for item in payload["candidates"])
    assert all(item["executed"] is False for item in payload["candidates"])
