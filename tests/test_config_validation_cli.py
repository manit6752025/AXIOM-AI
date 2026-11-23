from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from axiom.cli.main import app


runner = CliRunner()


def _validate(path):
    return runner.invoke(
        app,
        ["--no-banner", "config", "validate", "--path", str(path), "--json"],
    )


def _payload(result):
    assert result.stdout, result.output
    return json.loads(result.stdout)


def _assert_suggestions_only(payload):
    assert payload["schema_version"] == 1
    assert payload["valid"] is False
    assert payload["repair_applied"] is False
    assert payload["repair_mode"] == "suggestions_only"
    assert payload["suggestions"]


def test_config_validate_reports_missing_file_with_stable_code_and_no_write(tmp_path):
    path = tmp_path / "missing.yaml"

    result = _validate(path)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_not_found"
    assert not path.exists()


def test_config_validate_reports_malformed_yaml_without_echoing_values(tmp_path):
    path = tmp_path / "axiom.yaml"
    original = "model: [PRIVATE_CONFIG_VALUE\n"
    path.write_text(original, encoding="utf-8")

    result = _validate(path)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_invalid"
    assert "PRIVATE_CONFIG_VALUE" not in result.output
    assert path.read_text(encoding="utf-8") == original


def test_config_validate_rejects_oversized_file_with_bounded_no_write_read(tmp_path):
    path = tmp_path / "axiom.yaml"
    original = b"x: " + (b"a" * (256 * 1024 + 64))
    path.write_bytes(original)

    result = _validate(path)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_too_large"
    assert "256 KiB" in payload["message"]
    assert path.read_bytes() == original


def test_config_validate_rejects_symlink_without_reading_or_mutating_target(tmp_path):
    target = tmp_path / "target.yaml"
    target.write_text("project:\n  name: safe\n", encoding="utf-8")
    link = tmp_path / "linked.yaml"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")
    original = target.read_bytes()

    result = _validate(link)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_symlink_forbidden"
    assert target.read_bytes() == original
    assert link.is_symlink()


def test_config_validate_rejects_non_file_paths_with_stable_code(tmp_path):
    directory = tmp_path / "config-directory"
    directory.mkdir()

    result = _validate(directory)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_not_regular_file"
    assert directory.is_dir()


@pytest.mark.parametrize("contents", ["- one\n- two\n", "true\n", ""])
def test_config_validate_rejects_non_mapping_yaml(tmp_path, contents):
    path = tmp_path / "axiom.yaml"
    path.write_text(contents, encoding="utf-8")
    original = path.read_bytes()

    result = _validate(path)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_not_mapping"
    assert path.read_bytes() == original


@pytest.mark.parametrize("section_value", ["[]", "null", '"not-a-mapping"'])
def test_config_validate_rejects_wrong_shaped_known_sections(tmp_path, section_value):
    path = tmp_path / "axiom.yaml"
    original = f"model: {section_value}\n"
    path.write_text(original, encoding="utf-8")

    result = _validate(path)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    _assert_suggestions_only(payload)
    assert payload["error_code"] == "config_section_not_mapping"
    assert path.read_text(encoding="utf-8") == original


def test_config_validate_accepts_mapping_without_changing_the_file(tmp_path):
    path = tmp_path / "axiom.yaml"
    original = "project:\n  name: local-demo\nmodel:\n  path: ./models/qwen\n"
    path.write_text(original, encoding="utf-8")

    result = _validate(path)

    assert result.exit_code == 0, result.output
    payload = _payload(result)
    assert payload["schema_version"] == 1
    assert payload["status"] == "valid"
    assert payload["valid"] is True
    assert payload["error_code"] is None
    assert payload["suggestions"] == []
    assert payload["repair_applied"] is False
    assert payload["repair_mode"] == "suggestions_only"
    assert path.read_text(encoding="utf-8") == original
