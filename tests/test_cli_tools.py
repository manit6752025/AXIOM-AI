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
