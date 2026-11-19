import io
from pathlib import Path

from typer.testing import CliRunner

import axiom.cli.main as cli_main
from axiom.cli.startup import (
    AMBER,
    COFFEE,
    CREAM,
    render_startup,
    show_startup,
    suppress_for_arguments,
)


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_render_startup_contains_welcome_and_palette_copy():
    output = io.StringIO()

    render_startup(output, include_next_steps=True)

    rendered = output.getvalue()
    assert (
        "Know what fits before you train."
        in rendered
    )
    assert "Start here: " in rendered
    assert "axiom guide" in rendered
    assert "Next steps:" in rendered
    assert COFFEE == "#99684E"
    assert CREAM == "#DEC2AA"
    assert AMBER == "#171310"
    assert "38;2;222;194;170" in rendered
    assert "38;2;23;19;16" in rendered
    assert "38;2;153;104;78" in rendered


def test_show_startup_requires_an_interactive_tty(tmp_path: Path):
    output = io.StringIO()

    shown = show_startup(
        output,
        environ={"TERM_SESSION_ID": "test-session"},
        marker_dir=tmp_path,
    )

    assert shown is False
    assert output.getvalue() == ""
    assert not list(tmp_path.iterdir())


def test_show_startup_claims_one_marker_per_terminal_session(tmp_path: Path):
    environment = {"TERM_SESSION_ID": "test-session"}
    first_output = TTYBuffer()
    second_output = TTYBuffer()

    assert (
        show_startup(
            first_output,
            include_next_steps=True,
            environ=environment,
            marker_dir=tmp_path,
        )
        is True
    )
    assert (
        show_startup(
            second_output,
            environ=environment,
            marker_dir=tmp_path,
        )
        is False
    )

    assert "AXIOM" in first_output.getvalue()
    assert "Start here: " in first_output.getvalue()
    assert second_output.getvalue() == ""
    assert len(list(tmp_path.iterdir())) == 1


def test_show_startup_honors_no_banner_escape_hatch(tmp_path: Path):
    output = TTYBuffer()

    shown = show_startup(
        output,
        environ={"AXIOM_NO_BANNER": "1", "TERM_SESSION_ID": "test-session"},
        marker_dir=tmp_path,
    )

    assert shown is False
    assert output.getvalue() == ""
    assert not list(tmp_path.iterdir())


def test_show_startup_can_skip_machine_readable_output(tmp_path: Path):
    output = TTYBuffer()

    shown = show_startup(
        output,
        environ={"TERM_SESSION_ID": "test-session"},
        marker_dir=tmp_path,
        machine_readable=True,
    )

    assert shown is False
    assert output.getvalue() == ""
    assert not list(tmp_path.iterdir())


def test_machine_and_usage_arguments_are_suppressed():
    assert suppress_for_arguments(["--help"]) is True
    assert suppress_for_arguments(["--format=json"]) is True
    assert suppress_for_arguments(["--output-format=ndjson"]) is True
    assert suppress_for_arguments(["version"]) is False


def test_bare_axiom_prints_commands_and_starts_the_daemon(monkeypatch):
    calls: list[bool] = []

    def fake_show_startup(*, include_next_steps: bool = False, **kwargs):
        del kwargs
        calls.append(include_next_steps)
        return True

    monkeypatch.setattr(cli_main, "show_startup", fake_show_startup)
    monkeypatch.setattr(cli_main, "run_daemon", lambda: None)

    result = CliRunner().invoke(cli_main.app, [])

    assert result.exit_code == 0
    assert "Missing command" not in result.stdout
    assert calls == [False]
    assert "Useful commands" in result.stdout
    assert "axiom model inspect ./models/my-model" in result.stdout
    assert "Local daemon" in result.stdout


def test_no_banner_bare_axiom_still_starts_the_daemon(monkeypatch):
    startup_calls: list[bool] = []

    def fake_show_startup(*, include_next_steps: bool = False, **kwargs):
        del kwargs
        startup_calls.append(include_next_steps)
        return True

    monkeypatch.setattr(cli_main, "show_startup", fake_show_startup)
    monkeypatch.setattr(cli_main, "run_daemon", lambda: None)

    result = CliRunner().invoke(cli_main.app, ["--no-banner"])

    assert result.exit_code == 0
    assert startup_calls == []
    assert "Useful commands" in result.stdout


def test_daemon_json_option_passes_machine_readable_mode(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_daemon(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "run_daemon", fake_run_daemon)

    result = CliRunner().invoke(cli_main.app, ["daemon", "--json"])

    assert result.exit_code == 0, result.stdout
    assert captured["machine_readable"] is True
    assert result.stdout == ""
