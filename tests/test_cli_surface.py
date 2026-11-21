import json
from pathlib import Path

from typer.testing import CliRunner

import axiom.cli.main as cli_main

app = cli_main.app

runner = CliRunner()


def test_help_surfaces_first_run_commands_and_guide_if_available(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    help_result = runner.invoke(app, ["--help"])

    assert help_result.exit_code == 0, help_result.stdout
    assert "init" in help_result.stdout
    assert "project" in help_result.stdout

    guide_help_result = runner.invoke(app, ["guide", "--help"])
    if "guide" not in help_result.stdout.lower():
        assert guide_help_result.exit_code != 0
        return

    assert guide_help_result.exit_code == 0, guide_help_result.stdout
    guide_result = runner.invoke(app, ["guide"])

    assert guide_result.exit_code == 0, guide_result.stdout
    assert "AXIOM" in guide_result.stdout
    assert "local" in guide_result.stdout.lower()
    assert "axiom init" in guide_result.stdout or "axiom project" in guide_result.stdout
    assert "http://" not in guide_result.stdout.lower()
    assert "https://" not in guide_result.stdout.lower()


def test_shell_completion_can_be_viewed_for_supported_shells_without_installing(tmp_path: Path, monkeypatch):
    from typer.main import get_command

    monkeypatch.chdir(tmp_path)
    # Typer's built-in switches are still available for install/default viewing.
    command = get_command(app)
    option_flags = {
        option
        for parameter in command.params
        for option in (*getattr(parameter, "opts", ()), *getattr(parameter, "secondary_opts", ()))
    }
    assert "--install-completion" in option_flags
    assert "--show-completion" in option_flags

    markers = {
        # Click renamed its Bash completion variable between releases.
        "bash": ("_AXIOM_COMPLETE=complete_bash", "_AXIOM_COMPLETE=bash_complete"),
        "zsh": ("#compdef axiom",),
        # Click versions generate Fish shell functions and completion variables
        # with different names; either stable marker confirms a real script.
        "fish": ("function _axiom_completion", "_AXIOM_COMPLETE=fish_complete", "_AXIOM_COMPLETE=complete_fish", "__fish_complete_path"),
        "powershell": ("Register-ArgumentCompleter",),
    }
    for shell, accepted_markers in markers.items():
        result = runner.invoke(app, ["completion", shell])
        assert result.exit_code == 0, f"{shell}: {result.output}"
        assert any(marker in result.output for marker in accepted_markers), (
            f"{shell} completion script was not generated: {result.output[:500]}"
        )

    unsupported = runner.invoke(app, ["completion", "tcsh"])
    assert unsupported.exit_code != 0
    assert "Unsupported shell" in unsupported.output

    # Generating and previewing completion scripts must not modify shell
    # startup files, create workspace metadata, or leave files in the cwd.
    assert list(tmp_path.iterdir()) == []


def test_init_creates_a_project_that_project_commands_can_use(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(app, ["init", "first-project"])

    assert init_result.exit_code == 0, init_result.stdout
    project_root = tmp_path / "first-project"
    assert project_root.is_dir()
    assert (project_root / "axiom.yaml").is_file()
    assert (project_root / "README.md").is_file()
    for directory in (
        "data/raw",
        "data/processed",
        "models",
        "experiments",
        "evaluations",
        "outputs",
    ):
        assert (project_root / directory).is_dir()

    monkeypatch.chdir(project_root)

    info_result = runner.invoke(app, ["project", "info"])
    validate_result = runner.invoke(app, ["project", "validate"])

    assert info_result.exit_code == 0, info_result.stdout
    assert "Config:" in info_result.stdout
    assert "found" in info_result.stdout
    assert "README:" in info_result.stdout
    info_json = runner.invoke(app, ["project", "info", "--json"])
    assert info_json.exit_code == 0, info_json.stdout
    summary = json.loads(info_json.stdout)
    assert summary == {
        "schema_version": 1,
        "path": str(project_root),
        "config": {"path": "axiom.yaml", "exists": True},
        "readme": {"path": "README.md", "exists": True},
    }
    assert validate_result.exit_code == 0, validate_result.stdout
    assert "valid" in validate_result.stdout.lower()


def test_gui_port_uses_environment_default_and_explicit_flag_precedence(monkeypatch):
    monkeypatch.setenv("AXIOM_GUI_PORT", "40123")
    assert cli_main._resolve_gui_port(None) == 40123
    assert cli_main._resolve_gui_port(40234) == 40234


def test_gui_port_rejects_invalid_environment_configuration(monkeypatch):
    monkeypatch.setenv("AXIOM_GUI_PORT", "not-a-port")
    try:
        cli_main._resolve_gui_port(None)
    except ValueError as exc:
        assert "AXIOM_GUI_PORT" in str(exc)
    else:
        raise AssertionError("Invalid AXIOM_GUI_PORT must be rejected")

    monkeypatch.setenv("AXIOM_GUI_PORT", "65536")
    try:
        cli_main._resolve_gui_port(None)
    except ValueError as exc:
        assert "0 to 65535" in str(exc)
    else:
        raise AssertionError("Out-of-range AXIOM_GUI_PORT must be rejected")


def test_unified_llm_cli_exposes_min_p_for_local_generation():
    result = runner.invoke(app, ["llm", "chat", "--help"])
    assert result.exit_code == 0, result.stdout
    assert "--min-p" in result.stdout


def test_engine_fabric_cli_advertises_native_backend_and_sampler_contract():
    from typer.main import get_command

    root = get_command(app)
    engine = root.commands["engine"]
    route_chat = engine.commands["route-chat"]
    benchmark = engine.commands["route-benchmark"]

    backend_parameter = next(parameter for parameter in route_chat.params if "--backend" in getattr(parameter, "opts", ()))
    assert "axiom-native" in (backend_parameter.help or "")
    for command in (route_chat, engine.commands["auto-chat"], benchmark):
        flags = {
            option
            for parameter in command.params
            for option in (*getattr(parameter, "opts", ()), *getattr(parameter, "secondary_opts", ()))
        }
        for option in ("--top-p", "--top-k", "--min-p", "--repeat-penalty", "--seed", "--stop-sequence"):
            assert option in flags
    benchmark_backend = next(parameter for parameter in benchmark.params if "--backend" in getattr(parameter, "opts", ()))
    assert "axiom-native" in (benchmark_backend.help or "")


def test_global_cli_output_options_are_exposed_before_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--verbose" in result.output
    assert "--quiet" in result.output
    assert "--json-errors" in result.output


def test_quiet_mode_suppresses_normal_command_output():
    result = runner.invoke(app, ["--quiet", "version"])
    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_verbose_mode_keeps_normal_command_output():
    result = runner.invoke(app, ["--verbose", "version"])
    assert result.exit_code == 0, result.output
    assert "AXIOM" in result.output


def test_json_errors_emit_stable_machine_readable_envelope():
    import json

    result = runner.invoke(app, ["--json-errors", "completion", "tcsh"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {
        "schema_version": 1,
        "error": {
            "code": "AXIOM_CLI_ERROR",
            "message": "Unsupported shell. Choose bash, zsh, fish, or powershell.",
            "exit_code": 1,
        },
    }


def test_json_error_mode_uses_stable_code_for_invalid_global_options():
    import json

    result = runner.invoke(app, ["--json-errors", "--verbose", "--quiet", "version"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "invalid_global_options"
    assert payload["error"]["exit_code"] == 1


def test_json_error_option_can_be_combined_with_quiet_mode():
    import json

    result = runner.invoke(app, ["--quiet", "--json-errors", "completion", "tcsh"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "AXIOM_CLI_ERROR"



def _write_frontend_fixture(root: Path, installed_typescript: str) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "copy_webui.mjs").write_text("// fixture\\n", encoding="utf-8")
    (root / "package.json").write_text('{"name":"axiom-ui"}', encoding="utf-8")
    lock = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "dependencies": {"react": "^19.0.0"},
                "devDependencies": {"typescript": "~6.0.3", "typescript-eslint": "^8.73.0"},
            },
            "node_modules/react": {"version": "19.3.1"},
            "node_modules/typescript": {"version": "6.0.3"},
            "node_modules/typescript-eslint": {"version": "8.73.0"},
        },
    }
    (root / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    for name, version in (
        ("react", "19.3.1"),
        ("typescript", installed_typescript),
        ("typescript-eslint", "8.73.0"),
    ):
        package_dir = root / "node_modules" / name
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "package.json").write_text(
            json.dumps({"name": name, "version": version}),
            encoding="utf-8",
        )


def test_frontend_dependency_check_detects_typescript_lockfile_drift(tmp_path: Path):
    _write_frontend_fixture(tmp_path, "7.0.0")
    mismatches = cli_main._frontend_dependency_mismatches(tmp_path)
    assert mismatches == ["typescript: installed 7.0.0, locked 6.0.3"]


def test_gui_refresh_repairs_frontend_dependencies_before_lint_and_build(
    tmp_path: Path,
    monkeypatch,
):
    _write_frontend_fixture(tmp_path, "7.0.0")
    commands: list[list[str]] = []
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/npm")

    def fake_run(command, *, cwd, check):
        commands.append(command)
        if command[1] == "ci":
            package = tmp_path / "node_modules" / "typescript" / "package.json"
            package.write_text('{"name":"typescript","version":"6.0.3"}', encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    cli_main._refresh_control_center(tmp_path)

    assert commands == [
        ["/usr/bin/npm", "ci", "--include=dev", "--no-audit", "--no-fund"],
        ["/usr/bin/npm", "run", "lint"],
        ["/usr/bin/npm", "run", "build"],
    ]
    assert cli_main._frontend_dependency_mismatches(tmp_path) == []


def test_gui_refresh_skips_reinstall_when_dependencies_match_the_lockfile(
    tmp_path: Path,
    monkeypatch,
):
    _write_frontend_fixture(tmp_path, "6.0.3")
    commands: list[list[str]] = []
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_main.subprocess,
        "run",
        lambda command, *, cwd, check: (
            commands.append(command) or type("Completed", (), {"returncode": 0})()
        ),
    )

    cli_main._refresh_control_center(tmp_path)
    assert commands == [
        ["/usr/bin/npm", "run", "lint"],
        ["/usr/bin/npm", "run", "build"],
    ]


def test_gui_help_uses_the_public_axiom_executable_name():
    result = runner.invoke(app, ["gui", "--help"], prog_name="axiom")
    assert result.exit_code == 0, result.output
    assert "Usage: axiom gui" in result.output
    assert "python -m axiom.cli.main" not in result.output


def test_console_entrypoint_pins_the_public_program_name(monkeypatch):
    received: dict[str, str] = {}

    def fake_app(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli_main, "app", fake_app)
    cli_main.main()
    assert received == {"prog_name": "axiom"}
