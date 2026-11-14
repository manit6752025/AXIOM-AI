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
