import json
from pathlib import Path

from typer.testing import CliRunner

from axiom.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AXIOM" in result.stdout


def test_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Working tree" in result.stdout


def test_doctor_strict_treats_workspace_warnings_as_failures(tmp_path: Path):
    default_result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])
    strict_result = runner.invoke(app, ["doctor", "--strict", "--workspace", str(tmp_path)])

    assert default_result.exit_code == 0, default_result.stdout
    assert strict_result.exit_code == 1, strict_result.stdout
    assert "Strict mode" in strict_result.stdout


def test_doctor_strict_json_keeps_diagnostics_machine_readable(tmp_path: Path):
    result = runner.invoke(
        app,
        ["doctor", "--strict", "--workspace", str(tmp_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["workspace"]["summary"]["warnings"] > 0


def test_summary_json_describes_the_current_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["summary", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["project"]["detected"] is False
    assert payload["recommended"]["command"] == "axiom init my-ai"
    assert payload["hardware"]["cpu_cores"] >= 1


def test_validate_dataset(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"

    dataset.write_text(
        '{"instruction":"hello","output":"world"}\n'
        '{"instruction":"hello","output":"world"}\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["dataset", "validate", str(dataset)],
    )

    assert result.exit_code == 0
    assert "Duplicates" in result.stdout


def test_project_validate(tmp_path: Path, monkeypatch):
    for folder in (
        "data",
        "models",
        "experiments",
        "evaluations",
        "outputs",
    ):
        (tmp_path / folder).mkdir()

    (tmp_path / "axiom.yaml").write_text(
        "project:\n  name: test\n",
        encoding="utf-8",
    )

    (tmp_path / "README.md").write_text(
        "# AXIOM test\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["project", "validate"],
    )

    assert result.exit_code == 0


def test_module_entrypoint_registers_mcp_command():
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout


def test_config_validate_rejects_invalid_yaml_with_safe_repair_suggestions(tmp_path: Path):
    config = tmp_path / "axiom.yaml"
    original = "project: [\napi_token: DO_NOT_PRINT_THIS_SECRET\n"
    config.write_text(original, encoding="utf-8")

    result = runner.invoke(
        app,
        ["config", "validate", "--path", str(config), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["status"] == "invalid"
    assert payload["valid"] is False
    assert payload["error_code"] == "config_invalid"
    assert payload["repair_mode"] == "suggestions_only"
    assert payload["repair_applied"] is False
    assert payload["suggestions"]
    assert "DO_NOT_PRINT_THIS_SECRET" not in result.stdout
    assert config.read_text(encoding="utf-8") == original





def test_config_validate_suggests_safe_action_when_file_is_missing(tmp_path: Path):
    config = tmp_path / "missing.yaml"
    result = runner.invoke(app, ["config", "validate", "--path", str(config), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error_code"] == "config_not_found"
    assert payload["valid"] is False
    assert payload["repair_applied"] is False
    assert any("axiom init" in item for item in payload["suggestions"])
    assert not config.exists()


def test_config_validate_reports_wrong_top_level_shape_without_rewriting(tmp_path: Path):
    config = tmp_path / "axiom.yaml"
    original = "- one\n- two\n"
    config.write_text(original, encoding="utf-8")

    result = runner.invoke(app, ["config", "validate", "--path", str(config), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error_code"] == "config_not_mapping"
    assert payload["suggestions"]
    assert payload["repair_applied"] is False
    assert config.read_text(encoding="utf-8") == original


def test_config_validate_reports_oversized_file_without_rewriting(tmp_path: Path):
    config = tmp_path / "axiom.yaml"
    original = "x" * (256 * 1024 + 1)
    config.write_text(original, encoding="utf-8")

    result = runner.invoke(app, ["config", "validate", "--path", str(config), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error_code"] == "config_too_large"
    assert payload["repair_applied"] is False
    assert config.read_text(encoding="utf-8") == original


def test_config_validate_rejects_symlink_without_following_or_replacing_it(tmp_path: Path):
    import pytest

    target = tmp_path / "target.yaml"
    target.write_text("project:\n  name: do-not-change\n", encoding="utf-8")
    config = tmp_path / "axiom.yaml"
    try:
        config.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are unavailable on this platform: {exc}")

    result = runner.invoke(app, ["config", "validate", "--path", str(config), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error_code"] == "config_symlink_forbidden"
    assert payload["repair_applied"] is False
    assert config.is_symlink()
    assert target.read_text(encoding="utf-8") == "project:\n  name: do-not-change\n"


def test_config_validate_accepts_yaml_mapping(tmp_path: Path):
    config = tmp_path / "axiom.yaml"
    config.write_text("project:\n  name: demo\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["config", "validate", "--path", str(config)],
    )

    assert result.exit_code == 0
    assert "valid YAML" in result.stdout
