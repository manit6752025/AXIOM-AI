from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.testing import CliRunner

from axiom.cli.agent_profiles import register


def test_agent_profile_cli_json_create_and_list(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    app = typer.Typer()
    register(app)
    runner = CliRunner()

    listed = runner.invoke(app, ["agent-profile", "list", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)["profiles"][0]["name"] == "default"

    created = runner.invoke(app, ["agent-profile", "create", "Review team", "--capability", "dataset.profile", "--json"])
    assert created.exit_code == 0, created.stdout
    result = json.loads(created.stdout)
    assert result["task_capabilities"] == ["dataset.profile"]


def test_agent_profile_delete_requires_explicit_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    app = typer.Typer()
    register(app)
    result = CliRunner().invoke(app, ["agent-profile", "delete", "no-such-id"])
    assert result.exit_code == 2
    assert "without --yes" in result.stdout
