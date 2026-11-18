from __future__ import annotations

import json

from typer.testing import CliRunner

from axiom.cli.main import app


runner = CliRunner()


def _invoke_json_error(*args: str, expected_code: str, expected_exit: int = 1):
    result = runner.invoke(app, ["--no-banner", "--json-errors", *args])
    assert result.exit_code == expected_exit, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == expected_code
    assert payload["error"]["exit_code"] == expected_exit
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"]
    return payload


def test_workflow_command_errors_use_the_global_json_envelope():
    _invoke_json_error(
        "workflow", "list", "--search", "x" * 121,
        expected_code="workflow_error",
    )


def test_model_studio_command_errors_use_the_global_json_envelope(tmp_path):
    _invoke_json_error(
        "model", "studio", "convert-gguf",
        str(tmp_path / "missing-model"),
        "--llama-cpp-dir", str(tmp_path / "missing-llama"),
        "--output", str(tmp_path / "output.gguf"),
        expected_code="model_studio_error",
    )


def test_llm_validation_errors_use_the_global_json_envelope():
    _invoke_json_error(
        "llm", "chat", "local-model", "",
        expected_code="llm_error",
    )


def test_tool_catalog_errors_use_the_global_json_envelope():
    _invoke_json_error(
        "tools", "plan", "definitely-not-an-axiom-tool",
        expected_code="tool_not_found",
    )


def test_support_bundle_unsafe_output_preserves_exit_code_and_json_code(tmp_path):
    output = tmp_path / "bundle.zip"
    output.symlink_to(tmp_path / "target")
    _invoke_json_error(
        "support-bundle", "--output", str(output),
        expected_code="support_bundle_symlink",
        expected_exit=2,
    )


def test_human_error_mode_remains_readable_without_json_errors():
    result = runner.invoke(app, [
        "--no-banner", "workflow", "list", "--search", "x" * 121,
    ])
    assert result.exit_code == 1
    assert "Error: Workflow search" in result.output
    with __import__("contextlib").suppress(Exception):
        json.loads(result.output)
        raise AssertionError("Human mode unexpectedly returned JSON.")
