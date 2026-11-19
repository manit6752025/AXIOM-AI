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
