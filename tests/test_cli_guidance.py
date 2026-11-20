from pathlib import Path

from typer.testing import CliRunner

from axiom.cli.main import app

runner = CliRunner()


def test_guide_detects_complete_project_and_lists_next_commands(
    tmp_path: Path,
    monkeypatch,
):
    for folder in (
        "data",
        "models",
        "experiments",
        "evaluations",
        "outputs",
    ):
        (tmp_path / folder).mkdir()

    (tmp_path / "axiom.yaml").write_text(
        "project:\n  name: demo\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# demo\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["guide"])

    assert result.exit_code == 0
    assert "AXIOM is a local-first command-line tool" in result.stdout
    assert "AXIOM project: detected (axiom.yaml found)" in result.stdout
    assert "standard project paths are present" in result.stdout
    assert "axiom config validate" in result.stdout
    assert "axiom status" in result.stdout
    assert "axiom model list" in result.stdout


def test_guide_suggests_initialization_outside_project(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["guide"])

    assert result.exit_code == 0
    assert "AXIOM project: not detected" in result.stdout
    assert "axiom init my-ai" in result.stdout
    assert "cd my-ai" in result.stdout
    assert "axiom guide" in result.stdout


def test_check_reports_incomplete_project_without_network_or_backend_steps(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "axiom.yaml").write_text(
        "project:\n  name: demo\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0
    assert "AXIOM project detected" in result.stdout
    assert "Missing standard paths" in result.stdout
    assert "README.md" in result.stdout
    assert "axiom config validate" in result.stdout
    assert "http" not in result.stdout.lower()


def test_help_describes_local_observed_behavior():
    model_result = runner.invoke(app, ["model", "search", "--help"])
    project_result = runner.invoke(app, ["project", "validate", "--help"])

    assert model_result.exit_code == 0
    assert "local model paths" in model_result.stdout
    assert project_result.exit_code == 0
    assert "expected AXIOM project paths" in project_result.stdout


def test_gui_help_exposes_refresh_option():
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    assert "--refresh" in result.stdout
    assert "browser storage" in result.stdout


def test_refresh_control_center_runs_lint_then_build_without_deleting_ui(tmp_path, monkeypatch):
    import types
    import axiom.cli.main as main_cli

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/copy_webui.mjs").write_text("// copy", encoding="utf-8")
    (tmp_path / "axiom/webui").mkdir(parents=True)
    previous_index = tmp_path / "axiom/webui/index.html"
    previous_index.write_text("previous UI", encoding="utf-8")
    calls = []
    monkeypatch.setattr(main_cli.shutil, "which", lambda command: "/usr/bin/npm")

    def fake_run(args, *, cwd, check):
        calls.append((args, cwd, check))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(main_cli.subprocess, "run", fake_run)
    main_cli._refresh_control_center(tmp_path)

    assert [call[0][-1] for call in calls] == ["lint", "build"]
    assert all(call[1] == tmp_path and call[2] is False for call in calls)
    assert previous_index.read_text(encoding="utf-8") == "previous UI"


def test_update_cli_check_never_installs(monkeypatch):
    import axiom.cli.main as main_cli

    monkeypatch.setattr(main_cli.updater, "check_for_update", lambda current, channel: {
        "channel": channel, "current_version": current, "latest_version": "9.9.9",
        "update_available": True, "release": {"html_url": "https://example.invalid/release"},
    })
    monkeypatch.setattr(
        main_cli.updater,
        "download_and_install",
        lambda version: (_ for _ in ()).throw(AssertionError("check must not install")),
    )
    result = runner.invoke(app, ["--no-banner", "update", "--check"])
    assert result.exit_code == 0, result.output
    assert "Update available" in result.stdout
    assert "axiom update --auto" in result.stdout


def test_update_cli_auto_installs_without_prompt(monkeypatch):
    import axiom.cli.main as main_cli

    observed = []
    monkeypatch.setattr(main_cli.updater, "check_for_update", lambda current, channel: {
        "channel": channel, "current_version": current, "latest_version": "9.9.9",
        "update_available": True, "release": {"html_url": "https://example.invalid/release"},
    })
    monkeypatch.setattr(main_cli.updater, "download_and_install", lambda version: (
        observed.append(version) or {
            "installed_version": version, "user_data_preserved": True, "restart_required": True,
        }
    ))
    result = runner.invoke(app, ["--no-banner", "update", "--auto"])
    assert result.exit_code == 0, result.output
    assert observed == ["9.9.9"]
    assert "browser storage were left in place" in result.stdout
